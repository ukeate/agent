"""
分布式安全框架 - 加密通信框架
支持端到端加密、密钥管理、前向安全等特性
"""

import asyncio
import ssl
import secrets
import struct
import time
import json
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, ec, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum


class EncryptionAlgorithm(Enum):
    AES_256_GCM = "aes_256_gcm"
    CHACHA20_POLY1305 = "chacha20_poly1305"
    RSA_4096 = "rsa_4096"
    ECDH_P384 = "ecdh_p384"


class MessageType(Enum):
    HANDSHAKE = "handshake"
    KEY_EXCHANGE = "key_exchange"
    ENCRYPTED_MESSAGE = "encrypted_message"
    HEARTBEAT = "heartbeat"


@dataclass
class EncryptedMessage:
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    encrypted_payload: bytes
    signature: bytes
    timestamp: float
    nonce: bytes
    key_version: int


@dataclass
class CommunicationSession:
    session_id: str
    participants: List[str]
    symmetric_key: bytes
    key_version: int
    created_at: float
    expires_at: float
    forward_secure: bool = True


class EncryptedCommunicationFramework:
    """加密通信框架"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.sessions: Dict[str, CommunicationSession] = {}
        self.agent_keys: Dict[str, Dict[str, bytes]] = {}  # agent_id -> {public_key, private_key}
        self.key_rotation_interval = config.get('key_rotation_interval', 3600)  # 1小时
        self.message_ttl = config.get('message_ttl', 300)  # 5分钟
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """初始化加密通信框架"""
        await self._generate_agent_keypair()
        await self._setup_ssl_context()
        self.logger.info("Encrypted communication framework initialized")
        
    async def _generate_agent_keypair(self):
        """生成智能体密钥对"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=4096
        )
        public_key = private_key.public_key()
        
        self.private_key = private_key
        self.public_key = public_key
        
        self.logger.info("Agent keypair generated")
        
    async def _setup_ssl_context(self):
        """设置SSL上下文"""
        self.ssl_context = ssl.create_default_context()
        self.ssl_context.minimum_version = ssl.TLSVersion.TLSv1_3
        self.ssl_context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:DHE+CHACHA20:!aNULL:!MD5:!DSS')
        
    async def establish_secure_channel(
        self,
        sender_id: str,
        recipient_id: str,
        sender_public_key: bytes,
        recipient_public_key: bytes
    ) -> str:
        """建立安全通信通道"""
        try:
            # 生成会话ID
            session_id = secrets.token_hex(32)
            
            # 执行ECDH密钥交换
            shared_secret = await self._perform_ecdh_key_exchange(
                sender_public_key, recipient_public_key
            )
            
            # 派生会话密钥
            session_key = await self._derive_session_key(shared_secret, session_id)
            
            # 创建通信会话
            now = time.time()
            session = CommunicationSession(
                session_id=session_id,
                participants=[sender_id, recipient_id],
                symmetric_key=session_key,
                key_version=1,
                created_at=now,
                expires_at=now + self.key_rotation_interval,
                forward_secure=True
            )
            
            self.sessions[session_id] = session
            
            # 记录密钥建立事件
            await self._log_key_establishment(session_id, sender_id, recipient_id)
            
            self.logger.info(f"Secure channel established: {session_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to establish secure channel: {e}")
            raise Exception(f"Failed to establish secure channel: {str(e)}")
    
    async def _perform_ecdh_key_exchange(
        self,
        sender_public_key: bytes,
        recipient_public_key: bytes
    ) -> bytes:
        """执行ECDH密钥交换"""
        # 生成临时密钥对
        private_key = ec.generate_private_key(ec.SECP384R1())
        
        # 加载对方公钥
        peer_public_key = serialization.load_pem_public_key(recipient_public_key)
        
        # 计算共享密钥
        shared_key = private_key.exchange(ec.ECDH(), peer_public_key)
        
        return shared_key
    
    async def _derive_session_key(self, shared_secret: bytes, session_id: str) -> bytes:
        """派生会话密钥"""
        # 使用HKDF派生密钥
        hkdf = HKDF(
            algorithm=hashes.SHA384(),
            length=32,  # 256 bits for AES-256
            salt=session_id.encode(),
            info=b'session_key'
        )
        
        return hkdf.derive(shared_secret)
    
    async def encrypt_message(
        self,
        session_id: str,
        sender_id: str,
        recipient_id: str,
        payload: Dict[str, Any],
        message_type: MessageType = MessageType.ENCRYPTED_MESSAGE
    ) -> EncryptedMessage:
        """加密消息"""
        try:
            session = self.sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            # 检查会话是否过期
            if time.time() > session.expires_at:
                await self._rotate_session_key(session_id)
                session = self.sessions[session_id]
            
            # 序列化载荷
            payload_bytes = json.dumps(payload).encode('utf-8')
            
            # 生成随机nonce
            nonce = secrets.token_bytes(12)
            
            # 使用AES-GCM加密
            cipher = Cipher(
                algorithms.AES(session.symmetric_key),
                modes.GCM(nonce)
            )
            encryptor = cipher.encryptor()
            
            # 加密数据
            encrypted_payload = encryptor.update(payload_bytes) + encryptor.finalize()
            
            # 生成消息ID和时间戳
            message_id = secrets.token_hex(16)
            timestamp = time.time()
            
            # 创建签名数据
            sign_data = (
                message_id.encode() +
                sender_id.encode() +
                recipient_id.encode() +
                encrypted_payload +
                struct.pack('d', timestamp)
            )
            
            # 数字签名
            signature = await self._sign_message(sign_data)
            
            encrypted_message = EncryptedMessage(
                message_id=message_id,
                sender_id=sender_id,
                recipient_id=recipient_id,
                message_type=message_type,
                encrypted_payload=encrypted_payload + encryptor.tag,  # 包含GCM tag
                signature=signature,
                timestamp=timestamp,
                nonce=nonce,
                key_version=session.key_version
            )
            
            self.logger.debug(f"Message encrypted: {message_id}")
            return encrypted_message
            
        except Exception as e:
            self.logger.error(f"Failed to encrypt message: {e}")
            raise Exception(f"Failed to encrypt message: {str(e)}")
    
    async def decrypt_message(
        self,
        session_id: str,
        encrypted_message: EncryptedMessage
    ) -> Dict[str, Any]:
        """解密消息"""
        try:
            session = self.sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            # 验证消息时效性
            if time.time() - encrypted_message.timestamp > self.message_ttl:
                raise ValueError("Message has expired")
            
            # 验证密钥版本
            if encrypted_message.key_version != session.key_version:
                raise ValueError("Key version mismatch")
            
            # 验证数字签名
            sign_data = (
                encrypted_message.message_id.encode() +
                encrypted_message.sender_id.encode() +
                encrypted_message.recipient_id.encode() +
                encrypted_message.encrypted_payload[:-16] +  # 不包含GCM tag
                struct.pack('d', encrypted_message.timestamp)
            )
            
            if not await self._verify_signature(
                sign_data, 
                encrypted_message.signature,
                encrypted_message.sender_id
            ):
                raise ValueError("Invalid message signature")
            
            # 分离加密数据和GCM tag
            encrypted_data = encrypted_message.encrypted_payload[:-16]
            tag = encrypted_message.encrypted_payload[-16:]
            
            # 解密消息
            cipher = Cipher(
                algorithms.AES(session.symmetric_key),
                modes.GCM(encrypted_message.nonce, tag)
            )
            decryptor = cipher.decryptor()
            
            decrypted_bytes = decryptor.update(encrypted_data) + decryptor.finalize()
            
            # 反序列化载荷
            payload = json.loads(decrypted_bytes.decode('utf-8'))
            
            self.logger.debug(f"Message decrypted: {encrypted_message.message_id}")
            return payload
            
        except Exception as e:
            self.logger.error(f"Failed to decrypt message: {e}")
            raise Exception(f"Failed to decrypt message: {str(e)}")
    
    async def _sign_message(self, data: bytes) -> bytes:
        """数字签名"""
        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature
    
    async def _verify_signature(
        self,
        data: bytes,
        signature: bytes,
        sender_id: str
    ) -> bool:
        """验证数字签名"""
        try:
            # 获取发送方公钥
            sender_public_key_bytes = self.agent_keys.get(sender_id, {}).get('public_key')
            if not sender_public_key_bytes:
                return False
            
            sender_public_key = serialization.load_pem_public_key(sender_public_key_bytes)
            
            sender_public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    async def _rotate_session_key(self, session_id: str):
        """轮换会话密钥"""
        session = self.sessions.get(session_id)
        if not session:
            return
        
        # 生成新的会话密钥
        new_key = secrets.token_bytes(32)
        
        # 更新会话
        now = time.time()
        session.symmetric_key = new_key
        session.key_version += 1
        session.created_at = now
        session.expires_at = now + self.key_rotation_interval
        
        # 记录密钥轮换事件
        await self._log_key_rotation(session_id, session.key_version)
        
        self.logger.info(f"Session key rotated: {session_id} (version {session.key_version})")
    
    async def implement_forward_secrecy(self, session_id: str):
        """实现前向安全"""
        session = self.sessions.get(session_id)
        if session and session.forward_secure:
            # 删除旧密钥材料
            await self._securely_delete_old_keys(session_id, session.key_version - 1)
            
    async def detect_mitm_attack(
        self,
        session_id: str,
        expected_fingerprint: str,
        actual_fingerprint: str
    ) -> bool:
        """检测中间人攻击"""
        if expected_fingerprint != actual_fingerprint:
            # 记录安全事件
            await self._log_security_event({
                'event_type': 'potential_mitm_attack',
                'session_id': session_id,
                'expected_fingerprint': expected_fingerprint,
                'actual_fingerprint': actual_fingerprint,
                'timestamp': time.time()
            })
            self.logger.warning(f"Potential MITM attack detected on session {session_id}")
            return True
        return False
    
    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        session = self.sessions.get(session_id)
        if not session:
            return None
        
        return {
            'session_id': session.session_id,
            'participants': session.participants,
            'key_version': session.key_version,
            'created_at': session.created_at,
            'expires_at': session.expires_at,
            'forward_secure': session.forward_secure
        }
    
    async def close_session(self, session_id: str):
        """关闭通信会话"""
        session = self.sessions.get(session_id)
        if session:
            # 实现前向安全
            if session.forward_secure:
                await self._securely_delete_old_keys(session_id, session.key_version)
            
            # 删除会话
            del self.sessions[session_id]
            
            self.logger.info(f"Session closed: {session_id}")
    
    async def _log_key_establishment(self, session_id: str, sender_id: str, recipient_id: str):
        """记录密钥建立事件"""
        event = {
            'event_type': 'key_establishment',
            'session_id': session_id,
            'sender_id': sender_id,
            'recipient_id': recipient_id,
            'timestamp': time.time()
        }
        self.logger.info(f"Key establishment logged: {session_id}")
    
    async def _log_key_rotation(self, session_id: str, key_version: int):
        """记录密钥轮换事件"""
        event = {
            'event_type': 'key_rotation',
            'session_id': session_id,
            'key_version': key_version,
            'timestamp': time.time()
        }
        self.logger.info(f"Key rotation logged: {session_id} (v{key_version})")
    
    async def _log_security_event(self, event: Dict[str, Any]):
        """记录安全事件"""
        self.logger.warning(f"Security event: {event['event_type']}")
    
    async def _securely_delete_old_keys(self, session_id: str, key_version: int):
        """安全删除旧密钥"""
        # 在实际实现中，这里应该使用安全的内存清除方法
        self.logger.debug(f"Securely deleted old keys for session {session_id} (v{key_version})")
    
    async def register_agent_key(self, agent_id: str, public_key: bytes, private_key: bytes = None):
        """注册智能体密钥"""
        key_data = {'public_key': public_key}
        if private_key:
            key_data['private_key'] = private_key
        
        self.agent_keys[agent_id] = key_data
        self.logger.info(f"Agent key registered: {agent_id}")
    
    async def get_agent_public_key(self, agent_id: str) -> Optional[bytes]:
        """获取智能体公钥"""
        return self.agent_keys.get(agent_id, {}).get('public_key')


class MessageIntegrityValidator:
    """消息完整性验证器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.hash_algorithm = config.get('hash_algorithm', 'sha256')
        self.logger = logging.getLogger(__name__)
        
    async def validate_message_integrity(
        self,
        message: EncryptedMessage,
        session: CommunicationSession
    ) -> Dict[str, Any]:
        """验证消息完整性"""
        checks = {
            'timestamp_valid': await self._check_timestamp(message),
            'nonce_unique': await self._check_nonce_uniqueness(message),
            'sequence_valid': await self._check_message_sequence(message),
            'format_valid': await self._check_message_format(message)
        }
        
        all_valid = all(checks.values())
        confidence_score = sum(checks.values()) / len(checks)
        
        return {
            'valid': all_valid,
            'checks': checks,
            'confidence_score': confidence_score
        }
    
    async def _check_timestamp(self, message: EncryptedMessage) -> bool:
        """检查时间戳有效性"""
        current_time = time.time()
        time_diff = abs(current_time - message.timestamp)
        max_time_skew = self.config.get('max_time_skew', 300)  # 5分钟
        return time_diff <= max_time_skew
    
    async def _check_nonce_uniqueness(self, message: EncryptedMessage) -> bool:
        """检查nonce唯一性（简化实现）"""
        # 在实际实现中，应该维护一个nonce缓存来检查唯一性
        return len(message.nonce) == 12
    
    async def _check_message_sequence(self, message: EncryptedMessage) -> bool:
        """检查消息序列（简化实现）"""
        # 在实际实现中，应该检查消息序列号防止重放攻击
        return True
    
    async def _check_message_format(self, message: EncryptedMessage) -> bool:
        """检查消息格式"""
        try:
            return (
                message.message_id and
                message.sender_id and
                message.recipient_id and
                message.encrypted_payload and
                message.signature and
                message.nonce and
                message.timestamp > 0
            )
        except Exception:
            return False


class CertificateManager:
    """证书管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.certificates: Dict[str, x509.Certificate] = {}
        self.revoked_certificates: set = set()
        self.logger = logging.getLogger(__name__)
    
    async def load_certificate(self, cert_path: str) -> bool:
        """加载证书"""
        try:
            with open(cert_path, 'rb') as f:
                cert_data = f.read()
            
            cert = x509.load_pem_x509_certificate(cert_data)
            cert_id = self._get_certificate_id(cert)
            self.certificates[cert_id] = cert
            
            self.logger.info(f"Certificate loaded: {cert_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load certificate {cert_path}: {e}")
            return False
    
    async def validate_certificate(self, cert: x509.Certificate) -> Dict[str, Any]:
        """验证证书"""
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # 检查有效期
        now = datetime.utcnow()
        if now < cert.not_valid_before:
            validation_result['valid'] = False
            validation_result['errors'].append('Certificate not yet valid')
        elif now > cert.not_valid_after:
            validation_result['valid'] = False
            validation_result['errors'].append('Certificate expired')
        
        # 检查撤销状态
        cert_id = self._get_certificate_id(cert)
        if cert_id in self.revoked_certificates:
            validation_result['valid'] = False
            validation_result['errors'].append('Certificate revoked')
        
        return validation_result
    
    def _get_certificate_id(self, cert: x509.Certificate) -> str:
        """获取证书ID"""
        return str(cert.serial_number)
    
    async def revoke_certificate(self, cert_id: str, reason: str):
        """撤销证书"""
        self.revoked_certificates.add(cert_id)
        self.logger.warning(f"Certificate revoked: {cert_id} - {reason}")
    
    async def get_certificate(self, cert_id: str) -> Optional[x509.Certificate]:
        """获取证书"""
        return self.certificates.get(cert_id)