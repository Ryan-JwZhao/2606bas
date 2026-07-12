# BAS 局域网 HTTPS 反向代理

本目录提供 Nginx 的服务端部署配置，让手机通过 HTTPS 访问 BAS 的局域网 Web 控制页面和 PWA。

## 当前拓扑

```text
手机浏览器
    │ https://10.1.5.175/
    ▼
Nginx（服务端 IPv4 443）
    │ http://127.0.0.1:17070
    ▼
BAS WebControlServer
```

BAS 原有的 `http://10.1.5.175:17070` 不会被修改，普通用户仍然可以使用原来的 HTTP 入口。Nginx 当前只监听 `10.1.5.175:443`，IPv6 配置暂时保持注释状态。

## 部署步骤

1. 在运行 BAS 的服务端安装或放置 Nginx。仓库不包含 Nginx 二进制文件。
2. 确保 BAS 已启动 Web 控制，并监听 `0.0.0.0:17070` 或 `127.0.0.1:17070`。
3. 准备服务端证书。Nginx 必须有以下两个文件：

   ```text
   deploy/nginx/certs/bas-lan.crt
   deploy/nginx/certs/bas-lan.key
   ```

   如果服务端已有证书，直接放入上述路径即可。证书的 SAN 必须包含 `IP:10.1.5.175`。

   也可以在服务端使用 OpenSSL 生成临时自签名证书：

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\deploy\nginx\scripts\generate-self-signed-cert.ps1
   ```

   若 OpenSSL 不在 PATH 中：

   ```powershell
   powershell -ExecutionPolicy Bypass -File .\deploy\nginx\scripts\generate-self-signed-cert.ps1 -OpenSslPath C:\tools\openssl\bin\openssl.exe
   ```

4. 确认 Nginx 可执行文件在 PATH 中，或设置当前命令行的 `NGINX_EXE`：

   ```powershell
   $env:NGINX_EXE = 'C:\tools\nginx\nginx.exe'
   ```

5. 检查并启动 Nginx：

   ```powershell
   .\deploy\nginx\start-nginx.cmd
   ```

   配置变更后使用：

   ```powershell
   .\deploy\nginx\reload-nginx.cmd
   ```

   停止服务：

   ```powershell
   .\deploy\nginx\stop-nginx.cmd
   ```

6. 手机访问：

   ```text
   https://10.1.5.175/
   ```

## 重要限制

本配置只负责在服务端终止 TLS，不要求用户安装客户端证书，也不启用双向 TLS。

如果使用自签名或仅局域网 CA 证书，手机浏览器可能显示“证书不受信任”，并可能拒绝 Service Worker 或 PWA 安装。完全不让用户安装证书、同时不使用公网受信任证书时，无法保证浏览器把该内网 IP 识别为可安装 PWA 的安全来源；这是浏览器信任链限制，不是 Nginx 配置可以绕过的问题。

因此当前配置可以先用于验证 HTTPS 转发和视频流；若必须让 PWA 在用户侧无证书安装并稳定通过安全检查，需要后续提供浏览器默认信任的证书和匹配域名，或使用受设备管理策略信任的内部 CA。该方案不要求现在开放公网。

## 防火墙与验证

只需在服务端允许 TCP `443` 入站；BAS 的 `17070` 可仅供本机使用，也可以继续保留局域网 HTTP 入口。Windows 防火墙可在服务端管理员 PowerShell 中按需执行：

```powershell
New-NetFirewallRule -DisplayName "BAS Nginx HTTPS 443" -Direction Inbound -Protocol TCP -LocalPort 443 -Action Allow -Profile Private
```

服务端验证：

```powershell
Test-NetConnection 10.1.5.175 -Port 17070
Test-NetConnection 10.1.5.175 -Port 443
curl.exe -vk https://10.1.5.175/manifest.webmanifest
```

预期结果是 HTTPS 请求返回 `200`，并能读取 PWA manifest；自签名证书测试时 `curl` 需要 `-k`，浏览器仍可能显示证书警告。
