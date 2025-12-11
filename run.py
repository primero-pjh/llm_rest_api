"""HTTP/2 서버 실행 스크립트"""
import subprocess
import sys
import os


def generate_self_signed_cert():
    """개발용 자체 서명 인증서 생성"""
    cert_dir = os.path.dirname(os.path.abspath(__file__))
    cert_file = os.path.join(cert_dir, "cert.pem")
    key_file = os.path.join(cert_dir, "key.pem")

    if os.path.exists(cert_file) and os.path.exists(key_file):
        print("SSL 인증서가 이미 존재합니다.")
        return cert_file, key_file

    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization
        import datetime

        # 키 생성
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        # 인증서 생성
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME, "KR"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Seoul"),
            x509.NameAttribute(NameOID.LOCALITY_NAME, "Seoul"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Development"),
            x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        ])

        cert = (
            x509.CertificateBuilder()
            .subject_name(subject)
            .issuer_name(issuer)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(datetime.datetime.utcnow())
            .not_valid_after(datetime.datetime.utcnow() + datetime.timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                ]),
                critical=False,
            )
            .sign(key, hashes.SHA256(), default_backend())
        )

        # 파일 저장
        with open(key_file, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption()
            ))

        with open(cert_file, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))

        print(f"SSL 인증서 생성 완료: {cert_file}, {key_file}")
        return cert_file, key_file

    except ImportError:
        print("cryptography 패키지가 필요합니다: pip install cryptography")
        return None, None


def run_http2_server():
    """HTTP/2 서버 실행"""
    print("=" * 50)
    print("HTTP/2 서버 시작")
    print("=" * 50)

    # SSL 인증서 확인
    cert_file = "cert.pem"
    key_file = "key.pem"

    if os.path.exists(cert_file) and os.path.exists(key_file):
        # HTTPS + HTTP/2
        print("HTTPS (HTTP/2) 모드로 실행")
        print("URL: https://localhost:8000")
        print("Swagger: https://localhost:8000/docs")
        cmd = [
            sys.executable, "-m", "hypercorn",
            "main:app",
            "--bind", "0.0.0.0:8000",
            "--certfile", cert_file,
            "--keyfile", key_file,
            "--reload"
        ]
    else:
        # HTTP/2 cleartext (h2c)
        print("HTTP (h2c) 모드로 실행")
        print("URL: http://localhost:8000")
        print("참고: 브라우저에서 h2c 테스트는 제한적입니다.")
        print("curl 테스트: curl --http2-prior-knowledge http://localhost:8000/health")
        cmd = [
            sys.executable, "-m", "hypercorn",
            "main:app",
            "--bind", "0.0.0.0:8000",
            "--reload"
        ]

    print("=" * 50)
    subprocess.run(cmd)


if __name__ == "__main__":
    import ipaddress
    run_http2_server()
