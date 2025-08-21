# Kiến Trúc Tool Data Ingestion cho Enterprise Chatbot

## 1. Tổng Quan Kiến Trúc Tool

### 1.1 Mục Tiêu Chính
- Import và xử lý documents vào FAISS collections
- Tự động sinh metadata và document IDs
- Quản lý keywords và taxonomy
- Validate chất lượng dữ liệu trước khi ingest
- Monitoring và logging cho quá trình import

### 1.2 Kiến Trúc High-Level

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DATA INGESTION TOOL                             │
├─────────────────────────────────────────────────────────────────────┤
│  CLI Interface                Web Interface (Optional)              │
│  ├─ Batch Import             ├─ File Upload UI                      │
│  ├─ Single Document          ├─ Progress Tracking                   │
│  ├─ Validation Mode          ├─ Collection Management               │
│  └─ Status Commands          └─ Analytics Dashboard                 │
├─────────────────────────────────────────────────────────────────────┤
│                     PROCESSING ENGINE                               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │ Document Parser │  │ Content Chunker │  │ Metadata Engine │      │
│  │ - PDF/DOCX/MD   │  │ - Smart Split   │  │ - Auto Keywords │      │
│  │ - Text Extract  │  │ - Size Control  │  │ - Category Map  │      │
│  │ - Structure Det │  │ - Context Keep  │  │ - Doc ID Gen    │      │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘      │
│             │                   │                   │               │
│             └───────────────────┼───────────────────┘               │
│                                 │                                   │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │ Embedding Gen   │  │ Quality Control │  │Collection Router│      │
│  │ - Batch Process │  │ - Content Valid │  │ - Target Select │      │
│  │ - Vector Cache  │  │ - Duplicate Det │  │ - Load Balance  │      │
│  │ - Model Manage  │  │ - Score Metrics │  │ - Error Handle  │      │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘      │
├─────────────────────────────────────────────────────────────────────┤
│                      DATA STORAGE LAYER                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │ FAISS Manager   │  │ PostgreSQL Meta │  │ Redis Cache     │      │
│  │ - Collections   │  │ - Document Info │  │ - Temp Storage  │      │
│  │ - Vector Store  │  │ - Import Logs   │  │ - Progress Track│      │
│  │ - Index Update  │  │ - Quality Scores│  │ - Error Queue   │      │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 Cấu Trúc Project

```
data_ingestion_tool/
├── main.py                     # CLI entry point
├── config/
│   ├── __init__.py
│   ├── settings.py             # Load từ .env
│   └── collections.yaml        # Collection definitions
├── core/
│   ├── __init__.py
│   ├── document_parser.py      # Parse PDF/DOCX/MD files
│   ├── content_processor.py    # Chunk và process content
│   ├── metadata_engine.py      # Generate metadata/keywords
│   ├── embedding_generator.py  # Vector embeddings
│   └── collection_manager.py   # FAISS operations
├── models/
│   ├── __init__.py
│   ├── document.py            # Document data models
│   └── ingestion.py           # Import job models
├── utils/
│   ├── __init__.py
│   ├── file_handler.py        # File I/O operations
│   ├── validation.py          # Data quality checks
│   └── progress_tracker.py    # Progress monitoring
├── web/                       # Optional web interface
│   ├── __init__.py
│   ├── app.py                # FastAPI web interface
│   └── templates/            # HTML templates
├── tests/
│   ├── __init__.py
│   ├── test_parser.py
│   ├── test_processor.py
│   └── sample_documents/
├── requirements.txt
└── README.md
```

## 2. Chi Tiết Thiết Kế Components

### 2.1 Document ID Schema
```python
# Format: {collection_prefix}_{category}_{sequential_id}_{version}
# Examples:
PA_FEAT_001_v1    # Product A Features, document 1, version 1
PA_PRIC_002_v1    # Product A Pricing, document 2, version 1  
PB_FEAT_001_v2    # Product B Features, document 1, version 2
WARR_SUP_001_v1   # Warranty Support, document 1, version 1
CONT_INF_001_v1   # Contact Info, document 1, version 1
```

### 2.2 Metadata Structure
```python
{
    "doc_id": "PA_FEAT_001_v1",
    "title": "Product A Security Features",
    "collection": "product_a_features", 
    "category": "security",
    "product": "product_a",
    "keywords": ["security", "authentication", "encryption", "access_control"],
    "auto_keywords": ["bảo mật", "xác thực", "mã hóa"],
    "source_file": "product_a_security.pdf",
    "page_numbers": [1, 2, 3],
    "chunk_index": 0,
    "chunk_total": 5,
    "word_count": 245,
    "quality_score": 0.85,
    "language": "vi",
    "created_at": "2025-01-20T10:30:00Z",
    "updated_at": "2025-01-20T10:30:00Z",
    "version": 1,
    "tags": ["feature", "technical", "customer-facing"]
}
```

### 2.3 Collection Mapping Configuration
```yaml
# config/collections.yaml
collections:
  product_a_features:
    prefix: "PA_FEAT"
    description: "Product A Features and Capabilities"
    keywords_mapping:
      - "sản phẩm a"
      - "product a"
      - "tính năng"
      - "chức năng"
      - "khả năng"
    categories:
      - security
      - performance
      - integration
      - ui_ux
      - api
    max_chunk_size: 500
    overlap_size: 50
    
  product_a_pricing:
    prefix: "PA_PRIC"  
    description: "Product A Pricing and Plans"
    keywords_mapping:
      - "giá cả"
      - "pricing" 
      - "gói dịch vụ"
      - "chi phí"
    categories:
      - plans
      - pricing
      - billing
    max_chunk_size: 300
    overlap_size: 30
```

## 3. Core Processing Workflow

### 3.1 Document Ingestion Pipeline
```python
def process_document(file_path: str, target_collection: str) -> IngestionResult:
    """
    1. Parse Document → Extract text, structure, metadata
    2. Content Processing → Clean, normalize, detect language  
    3. Automatic Categorization → ML classification
    4. Keyword Extraction → Auto + manual keywords
    5. Content Chunking → Smart splitting with overlap
    6. Quality Validation → Scoring và filtering
    7. Embedding Generation → Vector creation
    8. Metadata Enrichment → Full metadata assembly
    9. Collection Storage → FAISS + PostgreSQL
    10. Verification → Post-ingest validation
    """
```

### 3.2 Batch Processing Strategy
- Process files in parallel (configurable workers)
- Progress tracking với Redis
- Error handling và retry logic
- Rollback capability for failed batches
- Memory optimization for large documents

## 4. CLI Interface Design

```bash
# Basic ingestion commands
python main.py ingest --file document.pdf --collection product_a_features
python main.py batch-ingest --dir ./documents --auto-route
python main.py ingest --file pricing.docx --collection product_a_pricing --category pricing

# Validation and testing
python main.py validate --file document.pdf --dry-run
python main.py test-embedding --text "sample query" --collection product_a_features

# Collection management
python main.py list-collections
python main.py collection-stats --name product_a_features
python main.py rebuild-collection --name product_a_features --confirm

# Quality control
python main.py quality-check --collection product_a_features --min-score 0.7
python main.py find-duplicates --collection all
python main.py optimize-keywords --collection product_a_features

# Monitoring and maintenance  
python main.py status
python main.py logs --tail 100
python main.py cleanup --remove-temp-files
```

## 5. Quality Control System

### 5.1 Content Quality Metrics
```python
quality_criteria = {
    "min_word_count": 50,           # Minimum words per chunk
    "max_word_count": 1000,         # Maximum words per chunk  
    "language_confidence": 0.8,      # Language detection confidence
    "content_coherence": 0.7,        # Semantic coherence score
    "keyword_relevance": 0.6,        # Keywords match content
    "duplicate_threshold": 0.95,     # Similarity threshold for duplicates
    "embedding_quality": 0.5         # Embedding generation success rate
}
```

### 5.2 Validation Rules
- File format support (PDF, DOCX, MD, TXT)
- Content language detection
- Minimum content length requirements
- Duplicate detection across collections
- Metadata completeness checks
- Embedding generation verification

## 6. Integration Points

### 6.1 Database Integration
- Load database config từ existing `.env`
- Extend PostgreSQL schema với ingestion tables
- Reuse Redis connection for caching
- Compatible với existing FAISS structure

### 6.2 Chatbot System Integration
- Same embedding model (sentence-transformers)
- Compatible collection structure
- Shared metadata format
- Real-time collection updates

## 7. Monitoring và Logging

### 7.1 Progress Tracking
- Real-time progress bars
- Batch processing status
- Error count và success rate
- Processing speed metrics
- Memory usage monitoring

### 7.2 Audit Trail
- Complete ingestion history
- Document versioning
- Change tracking
- Quality score evolution
- Performance benchmarks

## Đánh Giá Thiết Kế

**Ưu điểm**:
1. **Modular Architecture**: Tách biệt các concerns, dễ maintain
2. **Flexible Input**: Support multiple file formats
3. **Quality Control**: Built-in validation và scoring
4. **Integration Ready**: Compatible với existing chatbot system
5. **Scalable**: Batch processing với parallel workers
6. **Monitoring**: Comprehensive tracking và logging

**Cần Xem Xét**:
1. **Performance**: Large file processing strategy
2. **Memory Management**: Embedding generation for large batches
3. **Error Recovery**: Robust rollback mechanisms
4. **User Experience**: CLI vs Web interface preference
5. **Security**: File upload validation và sanitization

Thiết kế này có phù hợp với yêu cầu của bạn không? Bạn muốn tôi điều chỉnh hoặc mở rộng phần nào?