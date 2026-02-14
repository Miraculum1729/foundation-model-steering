# 完成推送到 GitHub

本地已准备好，commit 已创建。请在本机终端执行：

```bash
cd /mnt/hbnas/home/pfp/hiv2026/dplm/foundation-model-steering-temp
git push -u origin master
```

## 若推送时提示需要认证

**方式 1：SSH（若已配置 GitHub SSH 公钥）**
```bash
git remote set-url origin git@github.com:Miraculum1729/foundation-model-steering.git
git push -u origin master
```

**方式 2：Personal Access Token（HTTPS）**
- 在 GitHub → Settings → Developer settings → Personal access tokens 创建 token
- 推送时 Username 填你的 GitHub 用户名，Password 填 token

**方式 3：GitHub CLI**
```bash
gh auth login
git push -u origin master
```
