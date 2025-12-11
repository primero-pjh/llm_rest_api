"""Hypercorn 설정 파일 - HTTP/2 지원"""

# 바인딩 주소
bind = ["0.0.0.0:8000"]

# HTTP/2 활성화 (SSL 없이 h2c 사용)
h2_max_concurrent_streams = 100
h2_max_header_list_size = 65536
h2_max_inbound_frame_size = 16384

# Worker 설정
workers = 1

# 로깅
accesslog = "-"
errorlog = "-"
loglevel = "info"

# Keep-alive 설정
keep_alive_timeout = 120

# HTTP/2 cleartext (h2c) - SSL 없이 HTTP/2 사용
# 참고: 브라우저는 일반적으로 h2c를 지원하지 않음
# curl --http2-prior-knowledge 또는 SSL 사용 필요
