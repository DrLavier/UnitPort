"""Remote SSH server configuration for cloud-based Isaac Lab training.

Captures SSH connection parameters and remote Isaac Lab environment paths.
No network imports — this is pure config, mirroring isaac_lab_config.py's
data-only pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# Fields that are NEVER serialized to disk (security).
_SENSITIVE_FIELDS = frozenset({"password"})


@dataclass
class RemoteServerConfig:
    """SSH connection + remote Isaac Lab environment settings."""

    # --- identity ---
    name: str = ""                      # user-friendly label ("Lab GPU Server")

    # --- connection ---
    host: str = ""
    port: int = 22
    username: str = ""
    auth_method: str = "key"            # "key" | "password"
    private_key_path: str = ""          # local .pem / id_rsa path
    password: str = ""                  # runtime-only, never written to disk

    # --- remote environment (bare-metal mode) ---
    remote_isaac_lab_path: str = ""     # /home/user/IsaacLab
    remote_isaac_lab_launcher: str = "" # /home/user/IsaacLab/isaaclab.sh
    remote_python: str = ""             # fallback python path on server
    remote_work_dir: str = "/tmp/unitport_training"
    conda_env: str = ""                 # conda activate name (optional)

    # --- docker ---
    docker_enabled: bool = False
    docker_image: str = ""              # e.g. "nvcr.io/nvidia/isaac-lab:2.3.2"
    docker_container_name: str = ""     # e.g. "isaac-lab-5.1" (for docker exec)
    docker_host_data_dir: str = ""      # host mount source, e.g. "/home/trooperai/isaaclab_data"
    docker_container_data_dir: str = "/data"  # container mount target
    docker_extra_args: str = ""         # extra docker run/exec flags

    # --- transfer timeouts (seconds) ---
    upload_timeout: int = 120
    download_timeout: int = 300

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self, *, include_sensitive: bool = False) -> Dict[str, Any]:
        """Serialize to dict. Sensitive fields excluded by default."""
        d: Dict[str, Any] = {
            "name": self.name,
            "host": self.host,
            "port": self.port,
            "username": self.username,
            "auth_method": self.auth_method,
            "private_key_path": self.private_key_path,
            "remote_isaac_lab_path": self.remote_isaac_lab_path,
            "remote_isaac_lab_launcher": self.remote_isaac_lab_launcher,
            "remote_python": self.remote_python,
            "remote_work_dir": self.remote_work_dir,
            "conda_env": self.conda_env,
            "docker_enabled": self.docker_enabled,
            "docker_image": self.docker_image,
            "docker_container_name": self.docker_container_name,
            "docker_host_data_dir": self.docker_host_data_dir,
            "docker_container_data_dir": self.docker_container_data_dir,
            "docker_extra_args": self.docker_extra_args,
            "upload_timeout": self.upload_timeout,
            "download_timeout": self.download_timeout,
        }
        if include_sensitive:
            d["password"] = self.password
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RemoteServerConfig":
        """Deserialize from dict. Unknown keys are silently ignored."""
        return cls(
            name=str(data.get("name", "")),
            host=str(data.get("host", "")),
            port=int(data.get("port", 22)),
            username=str(data.get("username", "")),
            auth_method=str(data.get("auth_method", "key")),
            private_key_path=str(data.get("private_key_path", "")),
            password=str(data.get("password", "")),
            remote_isaac_lab_path=str(data.get("remote_isaac_lab_path", "")),
            remote_isaac_lab_launcher=str(data.get("remote_isaac_lab_launcher", "")),
            remote_python=str(data.get("remote_python", "")),
            remote_work_dir=str(data.get("remote_work_dir", "/tmp/unitport_training")),
            conda_env=str(data.get("conda_env", "")),
            docker_enabled=bool(data.get("docker_enabled", False)),
            docker_image=str(data.get("docker_image", "")),
            docker_container_name=str(data.get("docker_container_name", "")),
            docker_host_data_dir=str(data.get("docker_host_data_dir", "")),
            docker_container_data_dir=str(data.get("docker_container_data_dir", "/data")),
            docker_extra_args=str(data.get("docker_extra_args", "")),
            upload_timeout=int(data.get("upload_timeout", 120)),
            download_timeout=int(data.get("download_timeout", 300)),
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> Optional[str]:
        """Return an error message if the config is incomplete, or None."""
        if not self.name.strip():
            return "Server name is required."
        if not self.host.strip():
            return "Host is required."
        if not self.username.strip():
            return "Username is required."
        if self.auth_method == "key" and not self.private_key_path.strip():
            return "Private key path is required for key authentication."
        if self.auth_method == "password" and not self.password:
            return "Password is required for password authentication."
        if self.docker_enabled:
            if not self.docker_image.strip():
                return "Docker image is required when Docker mode is enabled."
            if not self.docker_host_data_dir.strip():
                return "Host data directory is required for Docker volume mount."
        else:
            if not self.remote_isaac_lab_path.strip():
                return "Remote Isaac Lab path is required."
        return None

    # ------------------------------------------------------------------
    # Docker helpers
    # ------------------------------------------------------------------

    @property
    def docker_use_exec(self) -> bool:
        """True if a container name is set → use `docker exec` on existing container."""
        return self.docker_enabled and bool(self.docker_container_name.strip())

    @property
    def effective_upload_dir(self) -> str:
        """Host directory where SFTP should upload training files.

        Docker mode: the host side of the volume mount (docker_host_data_dir).
        Bare-metal:  remote_work_dir.
        """
        if self.docker_enabled and self.docker_host_data_dir:
            return self.docker_host_data_dir.rstrip("/")
        return self.remote_work_dir.rstrip("/")
