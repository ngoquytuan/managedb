Tôi sẽ tạo script Python để nhập dữ liệu giả vào database và FAISS collections để test hệ thống. Script này sẽ tạo conversations, documents và analytics data realistic.

## data_seeder.py

```python
#!/usr/bin/env python3
"""
Data Seeder for Enterprise Chatbot
Tạo dữ liệu giả để test database và FAISS collections
"""

import asyncio
import os
import sys
import json
import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dataclasses import dataclass
import uuid

import psycopg2
from psycopg2.extras import RealDictCursor
import redis
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from engines.faiss_manager import FAISSCollectionManager, DocumentChunk
from models.schemas import PageContext

# Load environment variables
load_dotenv()

@dataclass
class FakeConversation:
    session_id: str
    user_message: str
    bot_response: str
    intent: str
    target_product: str
    confidence: float
    processing_time: float
    user_ip: str
    created_at: datetime

class DataSeeder:
    def __init__(self):
        self.db_url = os.getenv('DATABASE_URL')
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', 6379))
        
        if not self.db_url:
            raise ValueError("DATABASE_URL not found in environment variables")
        
        # Initialize connections
        self.redis_client = redis.Redis(
            host=self.redis_host,
            port=self.redis_port,
            decode_responses=True
        )
        
        self.faiss_manager = FAISSCollectionManager()
        
        print(f"✓ Database: {self.db_url}")
        print(f"✓ Redis: {self.redis_host}:{self.redis_port}")
    
    async def seed_all_data(self):
        """Seed tất cả dữ liệu test"""
        print("\n🚀 Bắt đầu seeding dữ liệu test...")
        
        try:
            # 1. Seed FAISS collections với documents
            await self.seed_faiss_documents()
            
            # 2. Seed database với conversations
            await self.seed_conversations()
            
            # 3. Seed Redis với cache data
            await self.seed_cache_data()
            
            # 4. Verify data
            await self.verify_data()
            
            print("\n✅ Hoàn thành seeding dữ liệu!")
            
        except Exception as e:
            print(f"\n❌ Lỗi khi seeding: {e}")
            raise
    
    async def seed_faiss_documents(self):
        """Seed FAISS collections với sample documents"""
        print("\n📚 Seeding FAISS documents...")
        
        # Initialize FAISS collections
        await self.faiss_manager.initialize_collections()
        
        # Sample documents cho từng collection
        documents_data = {
            'product_a_features': self.get_product_a_features_docs(),
            'product_a_pricing': self.get_product_a_pricing_docs(),
            'product_b_features': self.get_product_b_features_docs(),
            'warranty_support': self.get_warranty_support_docs(),
            'contact_company': self.get_contact_company_docs()
        }
        
        total_docs = 0
        for collection_name, docs in documents_data.items():
            print(f"  • Adding {len(docs)} documents to {collection_name}...")
            
            # Convert to DocumentChunk objects
            document_chunks = []
            for i, doc in enumerate(docs):
                chunk = DocumentChunk(
                    content=doc['content'],
                    metadata={
                        'title': doc.get('title', f'Document {i+1}'),
                        'category': doc.get('category', 'general'),
                        'product': doc.get('product'),
                        'section': doc.get('section'),
                        'source': doc.get('source', 'manual'),
                        'doc_id': f"{collection_name}_{i+1}"
                    }
                )
                document_chunks.append(chunk)
            
            # Add to FAISS collection
            added_count = await self.faiss_manager.add_documents_to_collection(
                collection_name, document_chunks
            )
            total_docs += added_count
            print(f"    ✓ Added {added_count} documents")
        
        print(f"📚 Total documents added: {total_docs}")
    
    async def seed_conversations(self):
        """Seed database với sample conversations"""
        print("\n💬 Seeding conversations...")
        
        conversations = self.generate_fake_conversations(500)  # 500 conversations
        
        conn = psycopg2.connect(self.db_url)
        cursor = conn.cursor()
        
        try:
            # Insert conversations
            insert_query = """
            INSERT INTO conversations 
            (session_id, user_message, bot_response, intent, target_product, 
             confidence, processing_time, sources_count, user_ip, created_at)
            VALUES (%(session_id)s, %(user_message)s, %(bot_response)s, %(intent)s, 
                    %(target_product)s, %(confidence)s, %(processing_time)s, 
                    %(sources_count)s, %(user_ip)s, %(created_at)s)
            """
            
            batch_size = 50
            for i in range(0, len(conversations), batch_size):
                batch = conversations[i:i + batch_size]
                batch_data = []
                
                for conv in batch:
                    batch_data.append({
                        'session_id': conv.session_id,
                        'user_message': conv.user_message,
                        'bot_response': conv.bot_response,
                        'intent': conv.intent,
                        'target_product': conv.target_product,
                        'confidence': conv.confidence,
                        'processing_time': conv.processing_time,
                        'sources_count': random.randint(1, 5),
                        'user_ip': conv.user_ip,
                        'created_at': conv.created_at
                    })
                
                cursor.executemany(insert_query, batch_data)
                conn.commit()
                print(f"  • Inserted batch {i//batch_size + 1}: {len(batch)} conversations")
            
            print(f"💬 Total conversations inserted: {len(conversations)}")
            
        finally:
            cursor.close()
            conn.close()
    
    async def seed_cache_data(self):
        """Seed Redis với sample cache data"""
        print("\n🗄️ Seeding cache data...")
        
        # Generate sample cache entries
        cache_entries = [
            {
                'key': 'chat:hash_abc123',
                'response': {
                    'response': 'Sản phẩm A có tính năng bảo mật SSL/TLS encryption, two-factor authentication và access control.',
                    'confidence': 0.95,
                    'intent': 'product_inquiry',
                    'sources': ['product_a_features_1', 'product_a_features_3']
                },
                'ttl': 3600
            },
            {
                'key': 'chat:hash_def456', 
                'response': {
                    'response': 'Giá cả sản phẩm A bắt đầu từ 99,000 VNĐ/tháng cho gói Basic.',
                    'confidence': 0.92,
                    'intent': 'pricing_inquiry',
                    'sources': ['product_a_pricing_1']
                },
                'ttl': 3600
            }
        ]
        
        for entry in cache_entries:
            self.redis_client.setex(
                entry['key'],
                entry['ttl'], 
                json.dumps(entry['response'])
            )
        
        # Add real-time metrics
        today = datetime.now().strftime('%Y-%m-%d')
        self.redis_client.setex(f"conversations:daily:{today}", 86400, "150")
        self.redis_client.setex(f"intents:daily:{today}:product_inquiry", 86400, "65")
        self.redis_client.setex(f"intents:daily:{today}:pricing_inquiry", 86400, "42")
        
        print(f"🗄️ Cache entries created: {len(cache_entries)}")
    
    def generate_fake_conversations(self, count: int) -> List[FakeConversation]:
        """Generate fake conversations data"""
        conversations = []
        
        # Sample user queries và responses
        queries_responses = [
            # Product inquiries
            ("Sản phẩm A có những tính năng gì?", "Sản phẩm A có các tính năng chính như bảo mật SSL, API integration, real-time analytics và scalable architecture.", "product_inquiry", "product_a"),
            ("Product B hoạt động như thế nào?", "Product B sử dụng AI engine để tự động hóa quy trình, với khả năng machine learning và predictive analytics.", "product_inquiry", "product_b"),
            ("Tính năng bảo mật của sản phẩm ra sao?", "Hệ thống bảo mật bao gồm encryption AES-256, two-factor authentication và security monitoring 24/7.", "product_inquiry", None),
            
            # Pricing inquiries  
            ("Giá cả sản phẩm A bao nhiêu?", "Sản phẩm A có 3 gói: Basic (99k/tháng), Pro (299k/tháng), Enterprise (999k/tháng).", "pricing_inquiry", "product_a"),
            ("Chi phí sử dụng thế nào?", "Chi phí được tính theo số user và tính năng sử dụng, bắt đầu từ 99,000 VNĐ/tháng.", "pricing_inquiry", None),
            ("Có khuyến mãi không?", "Hiện tại có ưu đãi 20% cho khách hàng đăng ký năm đầu và miễn phí trial 14 ngày.", "pricing_inquiry", None),
            
            # Support requests
            ("Làm sao để setup sản phẩm?", "Bạn có thể follow hướng dẫn setup trong docs hoặc liên hệ support team để được hỗ trợ chi tiết.", "support_request", None),
            ("Tôi gặp lỗi kết nối API", "Vui lòng kiểm tra API key và endpoint URL. Nếu vẫn lỗi, hãy gửi error logs cho team support.", "support_request", None),
            
            # Warranty inquiries
            ("Chính sách bảo hành như thế nào?", "Chúng tôi có chính sách bảo hành 12 tháng và hỗ trợ kỹ thuật 24/7 cho tất cả sản phẩm.", "warranty_inquiry", None),
            ("Có thể hoàn tiền không?", "Có chính sách hoàn tiền 100% trong 30 ngày đầu nếu không hài lòng với dịch vụ.", "warranty_inquiry", None),
            
            # Contact requests
            ("Thông tin liên hệ công ty?", "Công ty ABC, địa chỉ: 123 Nguyễn Trãi, Hà Nội. Hotline: 1900-xxxx, Email: support@company.com", "contact_request", None),
            ("Làm sao liên hệ support?", "Bạn có thể liên hệ qua hotline 1900-xxxx, email support@company.com hoặc chat trực tiếp trên website.", "contact_request", None),
        ]
        
        # Generate conversations
        for i in range(count):
            query, response, intent, product = random.choice(queries_responses)
            
            # Add variations
            if random.random() < 0.3:  # 30% chance để thêm variations
                query = f"{query} {random.choice(['Xin chào!', 'Cảm ơn bạn.', 'Vui lòng tư vấn.'])}"
            
            conv = FakeConversation(
                session_id=f"session_{uuid.uuid4().hex[:12]}",
                user_message=query,
                bot_response=response,
                intent=intent,
                target_product=product,
                confidence=random.uniform(0.7, 0.98),
                processing_time=random.uniform(0.5, 3.5),
                user_ip=f"192.168.{random.randint(1,255)}.{random.randint(1,255)}",
                created_at=datetime.now() - timedelta(
                    days=random.randint(0, 30),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )
            )
            conversations.append(conv)
        
        return conversations
    
    def get_product_a_features_docs(self) -> List[Dict]:
        """Sample documents cho Product A features"""
        return [
            {
                'content': 'Sản phẩm A cung cấp tính năng bảo mật SSL/TLS encryption đầu cuối, đảm bảo dữ liệu được mã hóa trong quá trình truyền tải. Hệ thống hỗ trợ chứng chỉ SSL wildcard và EV SSL cho độ tin cậy cao nhất.',
                'title': 'Tính năng bảo mật SSL',
                'category': 'security',
                'product': 'product_a',
                'section': 'features'
            },
            {
                'content': 'API Integration của sản phẩm A hỗ trợ RESTful API và GraphQL, cho phép tích hợp dễ dàng với các hệ thống third-party. Có rate limiting, authentication token và comprehensive documentation.',
                'title': 'API Integration',
                'category': 'integration',
                'product': 'product_a', 
                'section': 'features'
            },
            {
                'content': 'Real-time Analytics dashboard cung cấp insights về user behavior, performance metrics và business intelligence. Hỗ trợ custom reports, data export và real-time alerts.',
                'title': 'Analytics Dashboard',
                'category': 'analytics',
                'product': 'product_a',
                'section': 'features'
            },
            {
                'content': 'Scalable Architecture được thiết kế để handle millions of requests per day với auto-scaling capability. Sử dụng microservices pattern và container orchestration.',
                'title': 'Kiến trúc mở rộng',
                'category': 'architecture',
                'product': 'product_a',
                'section': 'features'
            },
            {
                'content': 'Two-Factor Authentication (2FA) bảo vệ accounts với multiple layers security. Hỗ trợ SMS, email và authenticator apps như Google Authenticator, Authy.',
                'title': 'Xác thực 2 lớp',
                'category': 'security',
                'product': 'product_a',
                'section': 'features'
            }
        ]
    
    def get_product_a_pricing_docs(self) -> List[Dict]:
        """Sample documents cho Product A pricing"""
        return [
            {
                'content': 'Gói Basic - 99,000 VNĐ/tháng: Phù hợp cho startup và small business. Bao gồm: 1,000 API calls/day, 5GB storage, email support, basic analytics. Không giới hạn số users.',
                'title': 'Gói Basic',
                'category': 'pricing',
                'product': 'product_a',
                'section': 'plans'
            },
            {
                'content': 'Gói Pro - 299,000 VNĐ/tháng: Dành cho medium business. Bao gồm: 10,000 API calls/day, 50GB storage, priority support, advanced analytics, custom integrations, SLA 99.9%.',
                'title': 'Gói Pro', 
                'category': 'pricing',
                'product': 'product_a',
                'section': 'plans'
            },
            {
                'content': 'Gói Enterprise - 999,000 VNĐ/tháng: Giải pháp cho large enterprise. Unlimited API calls, 500GB storage, 24/7 dedicated support, white-label option, custom development, SLA 99.99%.',
                'title': 'Gói Enterprise',
                'category': 'pricing', 
                'product': 'product_a',
                'section': 'plans'
            },
            {
                'content': 'Khuyến mãi đặc biệt: Giảm 20% cho khách hàng đăng ký trả trước 12 tháng. Miễn phí trial 14 ngày cho tất cả gói. Không phí setup và migration từ competitors.',
                'title': 'Ưu đãi và khuyến mãi',
                'category': 'promotion',
                'product': 'product_a', 
                'section': 'pricing'
            }
        ]
    
    def get_product_b_features_docs(self) -> List[Dict]:
        """Sample documents cho Product B features"""
        return [
            {
                'content': 'Product B sử dụng AI Engine tiên tiến với Machine Learning algorithms để tự động hóa business processes. Hỗ trợ neural networks, deep learning và natural language processing.',
                'title': 'AI Engine',
                'category': 'ai',
                'product': 'product_b',
                'section': 'features'
            },
            {
                'content': 'Predictive Analytics của Product B phân tích historical data để dự đoán trends và behaviors. Accuracy rate > 95% với real-time predictions và actionable insights.',
                'title': 'Predictive Analytics',
                'category': 'analytics',
                'product': 'product_b',
                'section': 'features'
            },
            {
                'content': 'Workflow Automation cho phép tạo custom workflows without coding. Drag-and-drop interface, conditional logic, integrations với 200+ third-party services.',
                'title': 'Tự động hóa quy trình',
                'category': 'automation',
                'product': 'product_b',
                'section': 'features'
            }
        ]
    
    def get_warranty_support_docs(self) -> List[Dict]:
        """Sample documents cho warranty & support"""
        return [
            {
                'content': 'Chính sách bảo hành 12 tháng cho tất cả sản phẩm và dịch vụ. Bảo hành cover tất cả technical issues, bugs và performance problems. Thời gian response < 24 hours.',
                'title': 'Chính sách bảo hành',
                'category': 'warranty',
                'section': 'support'
            },
            {
                'content': 'Hỗ trợ kỹ thuật 24/7 qua multiple channels: hotline, email, live chat và video call. Dedicated support team với experience > 5 years trong industry.',
                'title': 'Hỗ trợ kỹ thuật',
                'category': 'support',
                'section': 'support' 
            },
            {
                'content': 'Chính sách hoàn tiền 100% trong 30 ngày đầu nếu không hài lòng. Không cần lý do, process hoàn tiền trong 3-5 business days. Áp dụng cho tất cả gói dịch vụ.',
                'title': 'Hoàn tiền',
                'category': 'refund',
                'section': 'policy'
            }
        ]
    
    def get_contact_company_docs(self) -> List[Dict]:
        """Sample documents cho company contact"""
        return [
            {
                'content': 'Công ty ABC Technology - Chuyên cung cấp giải pháp công nghệ cho doanh nghiệp. Địa chỉ: Tầng 10, Tòa nhà XYZ, 123 Nguyễn Trãi, Thanh Xuân, Hà Nội. Website: www.company.com',
                'title': 'Thông tin công ty',
                'category': 'company',
                'section': 'contact'
            },
            {
                'content': 'Thông tin liên hệ: Hotline: 1900-1234 (24/7), Email: support@company.com, sales@company.com. Social media: Facebook/CompanyABC, LinkedIn/company-abc-tech',
                'title': 'Liên hệ',
                'category': 'contact',
                'section': 'contact'
            },
            {
                'content': 'Đội ngũ 50+ engineers và developers với kinh nghiệm 10+ năm. Đã phục vụ 500+ khách hàng từ startup đến enterprise. Established 2015, ISO 27001 certified.',
                'title': 'Về chúng tôi',
                'category': 'about',
                'section': 'company'
            }
        ]
    
    async def verify_data(self):
        """Verify dữ liệu đã được seed correctly"""
        print("\n🔍 Verifying seeded data...")
        
        # Check FAISS collections
        faiss_status = await self.faiss_manager.health_check()
        print(f"  • FAISS Collections: {faiss_status['collections']}")
        
        # Check database
        conn = psycopg2.connect(self.db_url)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM conversations")
        conv_count = cursor.fetchone()[0]
        print(f"  • Database conversations: {conv_count}")
        
        cursor.execute("SELECT intent, COUNT(*) FROM conversations GROUP BY intent")
        intent_stats = cursor.fetchall()
        print(f"  • Intent distribution: {dict(intent_stats)}")
        
        cursor.close()
        conn.close()
        
        # Check Redis
        cache_keys = self.redis_client.keys("chat:*")
        print(f"  • Redis cache entries: {len(cache_keys)}")
        
        print("✅ Data verification completed!")

async def main():
    """Main function"""
    print("🌱 Enterprise Chatbot Data Seeder")
    print("=" * 50)
    
    try:
        seeder = DataSeeder()
        await seeder.seed_all_data()
        
        print("\n🎉 Data seeding completed successfully!")
        print("\n📊 Data Summary:")
        print("  • FAISS: 5 collections với sample documents")
        print("  • Database: 500 fake conversations")
        print("  • Redis: Cache entries và real-time metrics")
        print("\n🚀 Bạn có thể bắt đầu test API endpoints!")
        
    except Exception as e:
        print(f"\n❌ Seeding failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
```

## Script chạy seeder

```bash
#!/bin/bash
# run_seeder.sh

echo "🌱 Starting data seeding process..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install dependencies if needed
pip install -r requirements.txt

# Run seeder
echo "🚀 Running data seeder..."
python data_seeder.py

echo "✅ Data seeding completed!"
```

## Test script để verify data

```python
# test_seeded_data.py
"""
Test script để verify seeded data hoạt động correctly
"""

import asyncio
import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

BASE_URL = f"http://localhost:{os.getenv('PORT', 8000)}"

async def test_api_endpoints():
    """Test các API endpoints với seeded data"""
    
    print("🧪 Testing API endpoints với seeded data...")
    
    # Test cases
    test_queries = [
        {
            "message": "Sản phẩm A có tính năng bảo mật gì?",
            "context": {
                "url": "http://localhost:3000/product-a/features",
                "title": "Product A Features",
                "product": "product_a",
                "section": "features"
            },
            "expected_intent": "product_inquiry"
        },
        {
            "message": "Giá cả sản phẩm A bao nhiêu?", 
            "context": {
                "url": "http://localhost:3000/product-a/pricing",
                "title": "Product A Pricing",
                "product": "product_a",
                "section": "pricing"
            },
            "expected_intent": "pricing_inquiry"
        },
        {
            "message": "Thông tin liên hệ công ty?",
            "context": {
                "url": "http://localhost:3000/contact",
                "title": "Contact Us",
                "section": "contact"
            },
            "expected_intent": "contact_request"
        }
    ]
    
    for i, test in enumerate(test_queries):
        print(f"\n📝 Test {i+1}: {test['message'][:30]}...")
        
        payload = {
            "message": test["message"],
            "session_id": f"test_session_{i+1}",
            "context": test["context"],
            "history": []
        }
        
        try:
            response = requests.post(
                f"{BASE_URL}/api/chat",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"  ✅ Response: {data['response'][:100]}...")
                print(f"  📊 Intent: {data.get('intent')} (expected: {test['expected_intent']})")
                print(f"  🎯 Confidence: {data.get('confidence', 0):.2f}")
                print(f"  ⏱️ Processing time: {data.get('processing_time', 0):.2f}s")
                print(f"  📚 Sources: {len(data.get('sources', []))}")
            else:
                print(f"  ❌ Error {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"  ❌ Request failed: {e}")
    
    # Test health endpoint
    print(f"\n🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/health")
        if response.status_code == 200:
            health = response.json()
            print(f"  ✅ System status: {health['status']}")
            print(f"  📊 Components: {list(health['components'].keys())}")
        else:
            print(f"  ❌ Health check failed: {response.status_code}")
    except Exception as e:
        print(f"  ❌ Health check error: {e}")

if __name__ == "__main__":
    asyncio.run(test_api_endpoints())
```

## Cách sử dụng:

1. **Chạy seeder**:
```bash
# Đảm bảo database và Redis đã chạy
docker-compose up -d postgres redis

# Chạy seeder
python data_seeder.py
```

2. **Start API server**:
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

3. **Test với dữ liệu giả**:
```bash
python test_seeded_data.py
```

Script này sẽ tạo:
- **FAISS Collections**: 5 collections với ~20 documents realistic về products, pricing, support
- **Database**: 500 conversations với distribution thực tế của intents
- **Redis Cache**: Sample cache entries và metrics

Dữ liệu được tạo sẽ có patterns giống real usage để test comprehensive các tính năng của chatbot system.