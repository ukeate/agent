"""
分布式安全框架 - 身份认证服务
支持PKI证书认证、OAuth2、多因子认证等多种认证方式
"""

from src.core.utils.timezone_utils import utc_now
import hmac
import jwt
import secrets
import hashlib
import time
import redis.asyncio as redis
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding

from src.core.logging import get_logger
class AuthenticationMethod(Enum):
    PKI_CERTIFICATE = "pki_cert"
    OAUTH2 = "oauth2"
    MULTI_FACTOR = "mfa"
    BIOMETRIC = "biometric"

class TrustLevel(Enum):
    UNTRUSTED = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class AgentIdentity:
    agent_id: str
    public_key: bytes
    certificate: Optional[x509.Certificate]
    trust_level: TrustLevel
    roles: List[str]
    attributes: Dict[str, Any]
    issued_at: datetime
    expires_at: datetime
    revoked: bool = False

@dataclass
class AuthenticationResult:
    authenticated: bool
    agent_identity: Optional[AgentIdentity]
    trust_score: float
    authentication_methods: List[AuthenticationMethod]
    session_token: Optional[str]
    error_message: Optional[str] = None

class IdentityAuthenticationService:
    """分布式身份认证服务"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.redis = None
        self.ca_certificates: Dict[str, x509.Certificate] = {}
        self.revoked_certificates: set = set()
        self.jwt_secret = config.get('jwt_secret', secrets.token_hex(32))
        self.session_timeout = config.get('session_timeout', 3600)
        self.logger = get_logger(__name__)
        
    async def initialize(self):
        """初始化认证服务"""
        self.redis = redis.from_url(self.config.get('redis_url', 'redis://localhost:6379'))
        await self._load_ca_certificates()
        await self._load_revocation_list()
        self.logger.info("Identity authentication service initialized")
        
    async def _load_ca_certificates(self):
        """加载CA证书"""
        ca_cert_paths = self.config.get('ca_certificates', [])
        for cert_path in ca_cert_paths:
            try:
                with open(cert_path, 'rb') as f:
                    cert_data = f.read()
                cert = x509.load_pem_x509_certificate(cert_data)
                subject_name = cert.subject.rfc4514_string()
                self.ca_certificates[subject_name] = cert
                self.logger.info(f"Loaded CA certificate: {subject_name}")
            except Exception as e:
                self.logger.error(f"Failed to load CA certificate {cert_path}: {e}")
    
    async def _load_revocation_list(self):
        """加载证书撤销列表"""
        try:
            revoked_list = await self.redis.smembers('revoked_certificates')
            self.revoked_certificates = {cert.decode() for cert in revoked_list}
            self.logger.info(f"Loaded {len(self.revoked_certificates)} revoked certificates")
        except Exception as e:
            self.logger.error(f"Failed to load revocation list: {e}")
    
    async def authenticate_agent(
        self, 
        agent_id: str,
        credentials: Dict[str, Any],
        authentication_methods: List[AuthenticationMethod]
    ) -> AuthenticationResult:
        """智能体身份认证"""
        try:
            authentication_results = []
            total_trust_score = 0.0
            
            for method in authentication_methods:
                if method == AuthenticationMethod.PKI_CERTIFICATE:
                    result = await self._authenticate_pki_certificate(
                        agent_id, credentials.get('certificate')
                    )
                elif method == AuthenticationMethod.OAUTH2:
                    result = await self._authenticate_oauth2(
                        agent_id, credentials.get('oauth_token')
                    )
                elif method == AuthenticationMethod.MULTI_FACTOR:
                    result = await self._authenticate_mfa(
                        agent_id, credentials.get('mfa_token')
                    )
                elif method == AuthenticationMethod.BIOMETRIC:
                    result = await self._authenticate_biometric(
                        agent_id, credentials.get('biometric_data')
                    )
                else:
                    continue
                
                authentication_results.append((method, result))
                if result['success']:
                    total_trust_score += result['trust_score']
            
            # 计算综合信任分数
            final_trust_score = min(total_trust_score, 1.0)
            min_trust_score = self.config.get('min_trust_score', 0.6)
            authenticated = final_trust_score >= min_trust_score
            
            if authenticated:
                # 创建智能体身份
                agent_identity = await self._create_agent_identity(
                    agent_id, credentials, final_trust_score
                )
                
                # 生成会话令牌
                session_token = await self._generate_session_token(agent_identity)
                
                return AuthenticationResult(
                    authenticated=True,
                    agent_identity=agent_identity,
                    trust_score=final_trust_score,
                    authentication_methods=authentication_methods,
                    session_token=session_token
                )
            else:
                return AuthenticationResult(
                    authenticated=False,
                    agent_identity=None,
                    trust_score=final_trust_score,
                    authentication_methods=authentication_methods,
                    session_token=None,
                    error_message="Trust score below threshold"
                )
                
        except Exception as e:
            self.logger.error(f"Authentication failed for agent {agent_id}: {e}")
            return AuthenticationResult(
                authenticated=False,
                agent_identity=None,
                trust_score=0.0,
                authentication_methods=authentication_methods,
                session_token=None,
                error_message=str(e)
            )
    
    async def _authenticate_pki_certificate(
        self, 
        agent_id: str, 
        certificate_data: bytes
    ) -> Dict[str, Any]:
        """PKI证书认证"""
        try:
            if not certificate_data:
                return {'success': False, 'trust_score': 0.0, 'error': 'No certificate provided'}
            
            # 解析证书
            cert = x509.load_pem_x509_certificate(certificate_data)
            
            # 验证证书有效期
            now = utc_now()
            if now < cert.not_valid_before or now > cert.not_valid_after:
                return {'success': False, 'trust_score': 0.0, 'error': 'Certificate expired'}
            
            # 检查撤销状态
            cert_serial = str(cert.serial_number)
            if cert_serial in self.revoked_certificates:
                return {'success': False, 'trust_score': 0.0, 'error': 'Certificate revoked'}
            
            # 验证证书链
            is_valid_chain = await self._verify_certificate_chain(cert)
            if not is_valid_chain:
                return {'success': False, 'trust_score': 0.0, 'error': 'Invalid certificate chain'}
            
            # 验证证书中的agent_id
            cert_agent_id = self._extract_agent_id_from_cert(cert)
            if cert_agent_id and cert_agent_id != agent_id:
                return {'success': False, 'trust_score': 0.0, 'error': 'Agent ID mismatch'}
            
            # 计算信任分数
            trust_score = await self._calculate_certificate_trust_score(cert)
            
            return {
                'success': True,
                'trust_score': trust_score,
                'certificate': cert,
                'public_key': cert.public_key().public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                )
            }
            
        except Exception as e:
            self.logger.error(f"PKI certificate authentication failed: {e}")
            return {'success': False, 'trust_score': 0.0, 'error': str(e)}
    
    async def _authenticate_oauth2(
        self, 
        agent_id: str, 
        oauth_token: str
    ) -> Dict[str, Any]:
        """OAuth2认证"""
        try:
            if not oauth_token:
                return {'success': False, 'trust_score': 0.0, 'error': 'No OAuth token provided'}
            
            # 验证OAuth2令牌
            token_info = await self._validate_oauth_token(oauth_token)
            if not token_info.get('valid', False):
                return {'success': False, 'trust_score': 0.0, 'error': 'Invalid OAuth token'}
            
            # 验证agent_id
            if token_info.get('sub') != agent_id:
                return {'success': False, 'trust_score': 0.0, 'error': 'Agent ID mismatch'}
            
            # 计算信任分数
            trust_score = min(0.8, token_info.get('trust_score', 0.5))
            
            return {
                'success': True,
                'trust_score': trust_score,
                'token_info': token_info
            }
            
        except Exception as e:
            self.logger.error(f"OAuth2 authentication failed: {e}")
            return {'success': False, 'trust_score': 0.0, 'error': str(e)}
    
    async def _authenticate_mfa(
        self, 
        agent_id: str, 
        mfa_token: str
    ) -> Dict[str, Any]:
        """多因子认证"""
        try:
            if not mfa_token:
                return {'success': False, 'trust_score': 0.0, 'error': 'No MFA token provided'}
            
            # 验证MFA令牌
            is_valid = await self._validate_mfa_token(agent_id, mfa_token)
            if not is_valid:
                return {'success': False, 'trust_score': 0.0, 'error': 'Invalid MFA token'}
            
            return {
                'success': True,
                'trust_score': 0.9  # MFA提供高信任度
            }
            
        except Exception as e:
            self.logger.error(f"MFA authentication failed: {e}")
            return {'success': False, 'trust_score': 0.0, 'error': str(e)}
    
    async def _authenticate_biometric(
        self, 
        agent_id: str, 
        biometric_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """生物特征认证"""
        try:
            if not biometric_data:
                return {'success': False, 'trust_score': 0.0, 'error': 'No biometric data provided'}
            
            # 验证生物特征数据
            match_score = await self._verify_biometric_data(agent_id, biometric_data)
            if match_score < 0.8:
                return {'success': False, 'trust_score': 0.0, 'error': 'Biometric verification failed'}
            
            return {
                'success': True,
                'trust_score': match_score
            }
            
        except Exception as e:
            self.logger.error(f"Biometric authentication failed: {e}")
            return {'success': False, 'trust_score': 0.0, 'error': str(e)}
    
    async def _verify_certificate_chain(self, cert: x509.Certificate) -> bool:
        """验证证书链"""
        try:
            issuer_name = cert.issuer.rfc4514_string()
            if issuer_name in self.ca_certificates:
                ca_cert = self.ca_certificates[issuer_name]
                ca_public_key = ca_cert.public_key()
                
                # 验证证书签名
                try:
                    ca_public_key.verify(
                        cert.signature,
                        cert.tbs_certificate_bytes,
                        padding.PKCS1v15(),
                        hashes.SHA256()
                    )
                    return True
                except Exception:
                    return False
            return False
        except Exception:
            return False
    
    def _extract_agent_id_from_cert(self, cert: x509.Certificate) -> Optional[str]:
        """从证书中提取智能体ID"""
        try:
            # 从Common Name或Subject Alternative Name中提取
            for attribute in cert.subject:
                if attribute.oid == x509.NameOID.COMMON_NAME:
                    return attribute.value
            return None
        except Exception:
            return None
    
    async def _calculate_certificate_trust_score(self, cert: x509.Certificate) -> float:
        """计算证书信任分数"""
        trust_score = 0.5  # 基础分数
        
        # 证书算法强度
        if cert.signature_algorithm_oid._name in ['sha256WithRSAEncryption', 'ecdsa-with-SHA256']:
            trust_score += 0.2
        
        # 证书有效期
        validity_period = cert.not_valid_after - cert.not_valid_before
        if validity_period.days <= 365:  # 一年内有效期
            trust_score += 0.1
        
        # 密钥长度
        public_key = cert.public_key()
        if hasattr(public_key, 'key_size') and public_key.key_size >= 2048:
            trust_score += 0.2
        
        return min(trust_score, 1.0)
    
    async def _validate_oauth_token(self, token: str) -> Dict[str, Any]:
        """验证OAuth2令牌"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=["HS256"])
        except Exception as e:
            return {"valid": False, "error": str(e)}

        sub = payload.get("sub") or payload.get("agent_id") or payload.get("user_id")
        if not sub:
            return {"valid": False, "error": "missing subject"}

        exp = payload.get("exp")
        return {"valid": True, "sub": str(sub), "trust_score": 0.8, "exp": exp}
    
    async def _validate_mfa_token(self, agent_id: str, token: str) -> bool:
        """验证MFA令牌（TOTP）"""
        if not (len(token) == 6 and token.isdigit()):
            return False

        secret = hmac.new(self.jwt_secret.encode(), agent_id.encode(), hashlib.sha1).digest()
        timestep = 30
        counter = int(time.time() // timestep)

        for drift in (-1, 0, 1):
            msg = (counter + drift).to_bytes(8, "big")
            digest = hmac.new(secret, msg, hashlib.sha1).digest()
            offset = digest[-1] & 0x0F
            code = (int.from_bytes(digest[offset : offset + 4], "big") & 0x7FFFFFFF) % 1000000
            if secrets.compare_digest(token, f"{code:06d}"):
                return True

        return False
    
    async def _verify_biometric_data(self, agent_id: str, data: Dict[str, Any]) -> float:
        """验证生物特征数据（挑战-响应）"""
        nonce = data.get("nonce")
        proof = data.get("proof")
        if not nonce or not proof:
            return 0.0

        secret = hmac.new(self.jwt_secret.encode(), agent_id.encode(), hashlib.sha256).digest()
        expected = hmac.new(secret, str(nonce).encode(), hashlib.sha256).hexdigest()
        return 1.0 if secrets.compare_digest(str(proof), expected) else 0.0
    
    async def _create_agent_identity(
        self,
        agent_id: str,
        credentials: Dict[str, Any],
        trust_score: float
    ) -> AgentIdentity:
        """创建智能体身份"""
        # 确定信任级别
        if trust_score >= 0.9:
            trust_level = TrustLevel.CRITICAL
        elif trust_score >= 0.8:
            trust_level = TrustLevel.HIGH
        elif trust_score >= 0.6:
            trust_level = TrustLevel.MEDIUM
        elif trust_score >= 0.4:
            trust_level = TrustLevel.LOW
        else:
            trust_level = TrustLevel.UNTRUSTED
        
        # 提取角色和属性
        roles = await self._extract_agent_roles(agent_id, credentials)
        attributes = await self._extract_agent_attributes(agent_id, credentials)
        
        now = utc_now()
        identity = AgentIdentity(
            agent_id=agent_id,
            public_key=credentials.get('public_key', b''),
            certificate=credentials.get('certificate'),
            trust_level=trust_level,
            roles=roles,
            attributes=attributes,
            issued_at=now,
            expires_at=now + timedelta(seconds=self.session_timeout)
        )
        
        # 缓存身份信息
        await self._cache_agent_identity(identity)
        
        return identity
    
    async def _generate_session_token(self, identity: AgentIdentity) -> str:
        """生成会话令牌"""
        payload = {
            'sub': identity.agent_id,
            'trust_level': identity.trust_level.value,
            'roles': identity.roles,
            'iat': int(identity.issued_at.timestamp()),
            'exp': int(identity.expires_at.timestamp()),
            'jti': secrets.token_hex(16)
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        
        # 缓存令牌
        await self.redis.setex(
            f"session_token:{identity.agent_id}",
            self.session_timeout,
            token
        )
        
        return token
    
    async def validate_session_token(self, token: str) -> Optional[AgentIdentity]:
        """验证会话令牌"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            agent_id = payload['sub']
            
            # 检查令牌是否在缓存中
            cached_token = await self.redis.get(f"session_token:{agent_id}")
            if not cached_token or cached_token.decode() != token:
                return None
            
            # 获取缓存的身份信息
            identity = await self._get_cached_agent_identity(agent_id)
            if identity and not identity.revoked:
                return identity
                
        except Exception as e:
            self.logger.error(f"Token validation error: {e}")
        
        return None
    
    async def revoke_agent_access(self, agent_id: str, reason: str):
        """撤销智能体访问权限"""
        # 标记身份为已撤销
        identity = await self._get_cached_agent_identity(agent_id)
        if identity:
            identity.revoked = True
            await self._cache_agent_identity(identity)
        
        # 删除会话令牌
        await self.redis.delete(f"session_token:{agent_id}")
        
        # 记录撤销事件
        await self._log_security_event({
            'event_type': 'access_revoked',
            'agent_id': agent_id,
            'reason': reason,
            'timestamp': utc_now().isoformat()
        })
        
        self.logger.info(f"Revoked access for agent {agent_id}: {reason}")
    
    async def _extract_agent_roles(self, agent_id: str, credentials: Dict[str, Any]) -> List[str]:
        """提取智能体角色"""
        # 从证书或其他凭据中提取角色信息
        return credentials.get('roles', ['agent'])
    
    async def _extract_agent_attributes(self, agent_id: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """提取智能体属性"""
        return {
            'agent_type': credentials.get('agent_type', 'standard'),
            'organization': credentials.get('organization', 'default'),
            'security_level': credentials.get('security_level', 'standard')
        }
    
    async def _cache_agent_identity(self, identity: AgentIdentity):
        """缓存智能体身份"""
        try:
            identity_data = {
                'agent_id': identity.agent_id,
                'trust_level': identity.trust_level.value,
                'roles': ','.join(identity.roles),
                'issued_at': identity.issued_at.isoformat(),
                'expires_at': identity.expires_at.isoformat(),
                'revoked': identity.revoked
            }
            
            await self.redis.hset(
                f"agent_identity:{identity.agent_id}",
                mapping=identity_data
            )
            await self.redis.expire(
                f"agent_identity:{identity.agent_id}",
                self.session_timeout
            )
        except Exception as e:
            self.logger.error(f"Failed to cache agent identity: {e}")
    
    async def _get_cached_agent_identity(self, agent_id: str) -> Optional[AgentIdentity]:
        """获取缓存的智能体身份"""
        try:
            identity_data = await self.redis.hgetall(f"agent_identity:{agent_id}")
            if not identity_data:
                return None
            
            # 解析缓存数据
            return AgentIdentity(
                agent_id=identity_data[b'agent_id'].decode(),
                public_key=b'',  # 不缓存敏感数据
                certificate=None,
                trust_level=TrustLevel(int(identity_data[b'trust_level'])),
                roles=identity_data[b'roles'].decode().split(',') if identity_data[b'roles'] else [],
                attributes={},
                issued_at=datetime.fromisoformat(identity_data[b'issued_at'].decode()),
                expires_at=datetime.fromisoformat(identity_data[b'expires_at'].decode()),
                revoked=identity_data[b'revoked'].decode().lower() == 'true'
            )
        except Exception as e:
            self.logger.error(f"Failed to get cached agent identity: {e}")
            return None
    
    async def _log_security_event(self, event: Dict[str, Any]):
        """记录安全事件"""
        try:
            event_key = f"security_event:{int(utc_now().timestamp())}:{secrets.token_hex(4)}"
            await self.redis.setex(event_key, 86400, str(event))  # 保存24小时
            self.logger.info(f"Security event logged: {event['event_type']}")
        except Exception as e:
            self.logger.error(f"Failed to log security event: {e}")

class ZeroTrustValidator:
    """零信任验证器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.risk_threshold = config.get('risk_threshold', 0.7)
        self.logger = get_logger(__name__)
        
    async def validate_agent_trust(
        self,
        agent_identity: AgentIdentity,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """零信任验证"""
        risk_factors = []
        risk_score = 0.0
        
        # 检查时间异常
        if await self._check_time_anomaly(agent_identity, context):
            risk_factors.append('unusual_access_time')
            risk_score += 0.2
        
        # 检查地理位置异常
        if await self._check_location_anomaly(agent_identity, context):
            risk_factors.append('unusual_location')
            risk_score += 0.3
        
        # 检查行为模式异常
        if await self._check_behavior_anomaly(agent_identity, context):
            risk_factors.append('unusual_behavior')
            risk_score += 0.4
        
        # 检查网络环境
        if await self._check_network_risk(context):
            risk_factors.append('high_risk_network')
            risk_score += 0.3
        
        trust_level = 'trusted' if risk_score < self.risk_threshold else 'untrusted'
        
        return {
            'trust_level': trust_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommended_actions': await self._get_recommended_actions(risk_score, risk_factors)
        }
    
    async def _check_time_anomaly(self, identity: AgentIdentity, context: Dict[str, Any]) -> bool:
        """检查时间异常"""
        current_hour = utc_now().hour
        # 非工作时间访问视为异常
        return current_hour < 6 or current_hour > 22
    
    async def _check_location_anomaly(self, identity: AgentIdentity, context: Dict[str, Any]) -> bool:
        """检查地理位置异常"""
        client_ip = context.get('client_ip')
        # 简化的地理位置检查
        if client_ip and not client_ip.startswith('192.168.'):
            return True
        return False
    
    async def _check_behavior_anomaly(self, identity: AgentIdentity, context: Dict[str, Any]) -> bool:
        """检查行为模式异常"""
        # 简化的行为异常检查
        request_rate = context.get('request_rate', 0)
        return request_rate > 100  # 每分钟超过100个请求
    
    async def _check_network_risk(self, context: Dict[str, Any]) -> bool:
        """检查网络环境风险"""
        user_agent = context.get('user_agent', '')
        # 检查可疑的User-Agent
        suspicious_agents = ['curl', 'wget', 'python-requests']
        return any(agent in user_agent.lower() for agent in suspicious_agents)
    
    async def _get_recommended_actions(self, risk_score: float, risk_factors: List[str]) -> List[str]:
        """获取推荐行动"""
        actions = []
        
        if risk_score >= 0.8:
            actions.append('立即阻断访问')
            actions.append('启动安全调查')
        elif risk_score >= 0.5:
            actions.append('增强监控')
            actions.append('要求额外验证')
        
        if 'unusual_access_time' in risk_factors:
            actions.append('验证访问时间合理性')
        
        if 'unusual_location' in risk_factors:
            actions.append('验证访问来源')
        
        return actions
