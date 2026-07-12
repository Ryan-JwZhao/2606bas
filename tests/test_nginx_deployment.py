from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]
NGINX_DIR = ROOT / "deploy" / "nginx"


def test_nginx_proxy_targets_bas_http_and_ipv4_listener() -> None:
    config = (NGINX_DIR / "nginx.conf").read_text(encoding="utf-8")

    assert "listen 10.1.5.175:443 ssl;" in config
    assert "server 127.0.0.1:17070;" in config
    assert "ssl_certificate certs/bas-lan.crt;" in config
    assert "ssl_certificate_key certs/bas-lan.key;" in config
    assert "# listen [fc00::10:872e:e311:780f:f456]:443 ssl;" in config


def test_nginx_proxy_preserves_streaming_and_pwa_routes() -> None:
    config = (NGINX_DIR / "nginx.conf").read_text(encoding="utf-8")

    assert "location = /mjpeg" in config
    assert "location = /api/mjpeg" in config
    assert "proxy_buffering off;" in config
    assert "proxy_read_timeout 1h;" in config
    assert "location /" in config
    assert "proxy_pass http://bas_web_control;" in config


def test_nginx_deployment_has_chinese_usage_readme_and_runtime_scripts() -> None:
    assert (NGINX_DIR / "readme.md").exists()
    assert (NGINX_DIR / "scripts" / "generate-self-signed-cert.ps1").exists()
    assert (NGINX_DIR / "start-nginx.cmd").exists()
    assert (NGINX_DIR / "reload-nginx.cmd").exists()
    assert (NGINX_DIR / "stop-nginx.cmd").exists()
