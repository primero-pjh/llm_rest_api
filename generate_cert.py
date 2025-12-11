"""개발용 자체 서명 SSL 인증서 생성 스크립트"""
import os
import ipaddress

try:
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    import datetime
except ImportError:
    print("cryptography 패키지를 설치해주세요:")
    print("pip install cryptography")
    exit(1)


def generate_certificate():
    cert_dir = os.path.dirname(os.path.abspath(__file__))
    cert_file = os.path.join(cert_dir, "cert.pem")
    key_file = os.path.join(cert_dir, "key.pem")

    print("개발용 자체 서명 SSL 인증서 생성 중...")

    # RSA 키 생성
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )

    # 인증서 정보
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "KR"),
        x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Seoul"),
        x509.NameAttribute(NameOID.LOCALITY_NAME, "Seoul"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Development"),
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
    ])

    # 인증서 생성
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
                x509.DNSName("127.0.0.1"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256(), default_backend())
    )

    # 개인키 저장
    with open(key_file, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption()
        ))
    print(f"개인키 생성: {key_file}")

    # 인증서 저장
    with open(cert_file, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    print(f"인증서 생성: {cert_file}")

    print("\n인증서 생성 완료!")
    print("이제 HTTP/2 서버를 실행할 수 있습니다:")
    print("  python run.py")
    print("\n또는 직접 실행:")
    print("  hypercorn main:app --certfile cert.pem --keyfile key.pem --reload")


if __name__ == "__main__":
    generate_certificate()
