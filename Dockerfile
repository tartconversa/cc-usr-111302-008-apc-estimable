FROM python:3.14-slim@sha256:9aaa474097d094040649f2162a627c0049b408f52138d980916788d80f04802f
WORKDIR /app
COPY . .
# 分析功能显式登记的数值依赖（见 pyproject.toml）；基础镜像 digest 保持固定
RUN pip install --no-cache-dir -e .
CMD ["python", "-I", "-m", "unittest", "discover", "-s", "tests", "-v"]
