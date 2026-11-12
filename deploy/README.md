# Deployment

Runs jobs defined in a manifest in parallel, one OpenStack VM per job. A job is just
`{name, script, config, args}` — `fleet.yml` doesn't hardcode a region, year, or script, so adding a
new job (e.g. a different year or region) is just adding an entry to the manifest and, if needed, a
new config file under `config/`.

---

## One-time setup

### 1. Install tools

```bash
pixi install -e deploy
pixi run -e deploy ansible-galaxy collection install -r deploy/requirements.yml
```

### 2. Configure OpenStack credentials

```bash
mkdir -p ~/.config/openstack
cp deploy/clouds.yaml.template ~/.config/openstack/clouds.yaml
# fill in auth_url, username, project details

# Store password separately (never committed)
cat > ~/.config/openstack/secure.yaml << EOF
clouds:
  openstack:
    auth:
      password: "your-password"
EOF
```

Verify connectivity:

```bash
pixi run -e deploy openstack image list
```

### 3. Configure S3 credentials

Both S3 stores (UZH Ceph and CDSE) authenticate with a named AWS profile, so no access key is
ever written into `.env` or into a VM's environment. Put the keys in `~/.aws/credentials`:

```ini
[uzh-glaciers]
aws_access_key_id = ...
aws_secret_access_key = ...

[cdse]
aws_access_key_id = ...
aws_secret_access_key = ...
```

```bash
chmod 700 ~/.aws && chmod 600 ~/.aws/credentials
```

`.env` names the profiles (`GLACIERS_S3_PROFILE`, `CDSE_S3_PROFILE`) and the endpoints. An
endpoint may instead come from the profile's `endpoint_url` in `~/.aws/config`. The playbooks render
exactly the profiles named by those `*_S3_PROFILE` entries onto each VM as
`/home/ubuntu/secrets/aws-credentials`. Any other profile you keep locally stays local. Adding a
third store therefore needs no deploy change — set its `*_S3_PROFILE` in `.env` and it follows.

### 4. Authenticate with GitHub

The playbook automatically retrieves your token and username via the GitHub CLI.
Make sure your authentication scopes include `read:packages` to access the GitHub Container Registry:

```bash
gh auth login --scopes read:packages
```

---

## Running

### Full production run

```bash
pixi run -e deploy ansible-playbook deploy/fleet.yml                                 # jobs-lrw.yml
pixi run -e deploy ansible-playbook deploy/fleet.yml -e "jobs_file=jobs-pass.yml"
```

`jobs_file` names a manifest next to `fleet.yml` and defaults to `jobs-lrw.yml`.
`redeploy.yml` takes the same variable — pass the one you deployed the fleet with.

### Single job or subset

```bash
pixi run -e deploy ansible-playbook deploy/fleet.yml -e "jobs=alps-2022,alps-2023"
```

`jobs` selects entries from the manifest by `name`, and each VM runs `python <script> <config> <args>`
from that entry unmodified. Processing parameters (scope, backscatter, coherence, LRW) live in the
config file. `args` carries the per-job OmegaConf overrides, which is how the annual Alps jobs share
one config and differ only by `year=<YYYY>`.

Ansible prints the IP and monitoring command for each VM at the end.

### Single script run (any script, auto-teardown)

Spins up one VM, runs the script, uploads logs to S3, then deletes the VM automatically:

```bash
pixi run -e deploy ansible-playbook deploy/run.yml \
  -e "cmd='scripts/s1-composites/process-composites.py config/lrw-alps.yml year=2023'"
```

The VM name is derived from the script filename (e.g. `glacier-run-process-composites`). Override with `-e "vm_name=my-job"`.

Like `fleet.yml`, this pushes `.env`, the AWS credentials file, the EE key and the ghcr.io login over SSH once the VM is up, rather than through cloud-init userdata — OpenStack keeps userdata in the instance metadata service, readable from the VM and through the API. The job then runs as a detached `glacier-run` systemd unit so it outlives the SSH session.

Ansible stays running and polls every 60 s until the script finishes (up to 12 h), then deletes the VM. Logs are uploaded to `s3://<bucket>/logs/` before shutdown. Follow progress with `sudo docker logs -f <vm-name>` or `sudo journalctl -u glacier-run -f` on the VM.

### Host pre-flight

`fleet.yml` and `run.yml` run `tasks/prepare-host.yml` on every VM before anything else touches
Docker: it masks the apt timers, waits for any unattended-upgrade already in flight, and installs
Docker if the image lacks it.

`apt-daily-upgrade.timer` is `Persistent=true` with a 60-minute randomized delay, so on an image
that has missed its 06:00 trigger it fires shortly after boot — every VM starts an unattended
upgrade at an unpredictable moment within an hour of coming up. needrestart then restarts every
daemon linked against the new libc6/libsystemd0, containerd and sshd included, which kills an
in-flight `docker pull` (`failed to copy: failed to send write: EOF`) and closes port 22 for up to a
minute (`Connection refused`). Because the delay is randomized per VM, a different subset of the
fleet failed on every run. Masking the timers removes that race.

Hosts are patched by refreshing the image named by `openstack_image`, not on every boot: the image
is a snapshot of a finished fleet VM with Docker emptied, `apt dist-upgrade` applied, the timers
masked, and the secrets, logs and cloud-init state removed. Upgrading at boot instead added 15–30
minutes per VM (unpacking a kernel's headers alone takes over 20 on these disks) and gave VMs
launched on different days different packages. To drain the pending upgrades on a VM anyway —
service restarts blocked, then a reboot into the new packages — pass `-e host_upgrade=true`.

`redeploy.yml` includes the same file with `host_upgrade: false`: those VMs are running jobs, so it
only disarms the timers and reconciles Docker, never reboots.

### Roll out a new image to a running fleet

Stops the container on every existing `glacier-fleet-*` VM, re-syncs `.env`, the AWS credentials file and the EE service account key from your local machine, pulls the latest image, and restarts the container with the same script/config it was launched with (looked up from the jobs manifest by VM name) — without recreating any VMs:

```bash
pixi run -e deploy ansible-playbook deploy/redeploy.yml
```

Use this after pushing a new image build to pick it up on VMs from an already-running `fleet.yml` deploy.

---

## Docker: pull and run locally

### Login to ghcr.io

```bash
echo $(gh auth token) | docker login ghcr.io -u $(gh api user --jq .login) --password-stdin
```

`.env` carries no S3 keys, so every invocation also needs the credentials file mounted. The image
sets `AWS_SHARED_CREDENTIALS_FILE=/app/secrets/aws-credentials`, so mounting there is enough —
the container runs as root, and relying on `~/.aws` would mean `/root/.aws`.

### Run a script non-interactively

```bash
docker run --rm --env-file .env \
  -v ~/.aws/credentials:/app/secrets/aws-credentials:ro \
  ghcr.io/lqgentner/glace-production:latest \
  python scripts/s1-composites/process-composites.py config/lrw-alps.yml year=2023
```

### Run interactively (bash shell)

```bash
docker run -it --rm --env-file .env \
  -v ~/.aws/credentials:/app/secrets/aws-credentials:ro \
  ghcr.io/lqgentner/glace-production:latest bash
```

### Mount local workspace (for development)

```bash
docker run -it --rm --env-file .env \
  -v ~/.aws/credentials:/app/secrets/aws-credentials:ro \
  -v "$(pwd)":/app \
  ghcr.io/lqgentner/glace-production:latest bash
```

---

## Monitoring

### Manual

SSH into a VM and follow logs in real time:

```bash
# Via Docker (live stream)
ssh ubuntu@<vm-ip> 'sudo docker logs -f glacier-fleet-<job-name>'

# Via log file (persists after container exits)
ssh ubuntu@<vm-ip> 'tail -f /home/ubuntu/logs/glacier-<scope>-<year>-<aggregation>.log'
```

Check if the job is still running:

```bash
ssh ubuntu@<vm-ip> 'sudo docker ps'
```

Copy the log file back to your machine:

```bash
scp ubuntu@<vm-ip>:/home/ubuntu/logs/glacier-alps-total-2022-median.log ./logs/
```

---

## Teardown

Run manually once you are done inspecting the VMs:

```bash
pixi run -e deploy ansible-playbook deploy/teardown.yml
```

To delete only a subset:

```bash
pixi run -e deploy ansible-playbook deploy/teardown.yml -e "jobs=alps-2022,alps-2023"
```
