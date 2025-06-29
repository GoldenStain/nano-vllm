# Pre-commit 设置说明

## 安装 pre-commit

```bash
# 安装 pre-commit
pip install pre-commit

# 在项目根目录安装 git hooks
pre-commit install
```

## 使用方法

### 自动运行（推荐）
安装后，每次 `git commit` 时会自动运行代码检查和格式化。

### 手动运行
```bash
# 对所有文件运行检查
pre-commit run --all-files

# 只对暂存的文件运行检查
pre-commit run

# 运行特定的hook
pre-commit run black
pre-commit run isort
```

## 配置说明

- **isort**: 自动排序 import 语句，使用 black 兼容模式
- **black**: 自动格式化代码，行长度88字符
- **其他检查**: 移除空格、检查文件语法等

## 跳过检查（不推荐）
如果需要跳过检查提交：
```bash
git commit --no-verify -m "commit message"
```
