# Benchmark hook: when $BENCH_PROXY is set, this Python's HTTPS goes through the
# recording proxy (both libraries read these standard variables).
import os
if os.environ.get("BENCH_PROXY"):
    os.environ["HTTPS_PROXY"] = os.environ["HTTP_PROXY"] = os.environ["BENCH_PROXY"]
    os.environ["https_proxy"] = os.environ["http_proxy"] = os.environ["BENCH_PROXY"]
    os.environ["SSL_CERT_FILE"] = os.environ["REQUESTS_CA_BUNDLE"] = os.environ["BENCH_CA"]
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "localhost,127.0.0.1"
