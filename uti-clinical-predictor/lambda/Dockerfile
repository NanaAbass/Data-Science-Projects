# ── Step 1: Base image ───────────────────────────────────────────────────────
# AWS provides an official Lambda base image for Python 3.11.
# It includes the Lambda Runtime Interface Client (RIC) which handles
# the communication between Lambda's execution environment and your handler.
FROM public.ecr.aws/lambda/python:3.11

# ── Step 2a: Install C build tools ───────────────────────────────────────────
# The Lambda base image ships without gcc/g++. Packages like scipy and numpy
# fall back to building from source when a pre-built wheel is not available.
# Installing the toolchain first ensures those builds succeed.
# yum clean all removes the cache afterwards to keep the layer lean.
RUN yum install -y gcc gcc-c++ && yum clean all

# ── Step 2b: Install Python dependencies ─────────────────────────────────────
# Copy requirements first (before app code) so Docker can cache this layer.
# If only handler.py changes on rebuild, this pip install layer is reused.
# --prefer-binary tells pip to use pre-built wheels wherever available,
# avoiding source compilation entirely for scipy, numpy, and sklearn.
COPY requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# ── Step 3: Copy application code ────────────────────────────────────────────
# Copy the handler into the Lambda task root directory.
# LAMBDA_TASK_ROOT is set by the base image to /var/task.
COPY handler.py ${LAMBDA_TASK_ROOT}/handler.py

# ── Step 4: Set the handler entrypoint ───────────────────────────────────────
# Tells Lambda which function to invoke: <module_name>.<function_name>
# handler.py → lambda_handler function.
CMD ["handler.lambda_handler"]
