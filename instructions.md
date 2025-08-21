# Kiến Trúc Tool Quản Lý Dữ liệu - Document Ingestion System

## 1. Tổng Quan Kiến Trúc

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA INGESTION TOOL                          │
├─────────────────────────────────────────────────────────────────────────┤
│  CLI Interface (Click) + Web Interface (FastAPI/Streamlit)             │
│  ├── Document Parser (PDF, DOCX, TXT, MD)                              │
│  ├── Content Chunker & Metadata Extractor                              │
│  └── Progress Tracking & Validation                                     │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────────┐
│                    PROCESSING PIPELINE                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  Document Analyzer                                                      │
│  ├── Auto-categorization (ML classifier)                               │
│  ├── Keyword extraction (TF-IDF + LLM)                                 │
│  ├── Collection assignment logic                                       │
│  └── Document ID generation                                            │
└─────────────────────┬───────────────────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────────────────┐
│                     DATA STORAGE                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  FAISS Collections        │    PostgreSQL Metadata                    │
│  ├── Vector embeddings    │    ├── Document registry                  │
│  ├── Collection indices   │    ├── Chunk relationships               │
│  └── Search optimization  │    └── Processing logs                   │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. Cấu Trúc Project

```
data_ingestion_tool/
├── main.py                    # CLI entry point
├── web_interface.py           # Web UI với Streamlit  
├── requirements.txt           # Dependencies
├── config/
│   ├── __init__.py
│   ├── settings.py           # Load .env configuration
│   └── collection_config.py  # FAISS collection definitions
├── core/
│   ├── __init__.py
│   ├── document_parser.py    # Parse multiple file formats
│   ├── content_processor.py  # Chunking và metadata extraction
│   ├── collection_manager.py # FAISS operations
│   └── metadata_manager.py   # PostgreSQL operations
├── analyzers/
│   ├── __init__.py
│   ├── categorizer.py       # Auto-categorization
│   ├── keyword_extractor.py # Keyword extraction
│   └── quality_checker.py   # Content quality validation
├── utils/
│   ├── __init__.py
│   ├── file_utils.py        # File operations
│   ├── progress_tracker.py  # Progress tracking
│   └── validation.py        # Data validation
├── templates/              # Document templates
│   ├── product_a_template.json
│   ├── pricing_template.json
│   └── support_template.json
└── data/
    ├── input/              # Input documents
    ├── processed/          # Processed files
    └── logs/              # Processing logs
```

## 3. Configuration System

### config/settings.py
```python
"""
Configuration management với .env support
"""
import os
from typing import Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class DatabaseConfig:
    """Database connection settings"""
    url: str
    host: str
    port: int
    name: str
    user: str
    password: str

@dataclass
class RedisConfig:
    """Redis connection settings"""
    host: str
    port: int
    db: int = 0

@dataclass
class LLMConfig:
    """LLM provider settings"""
    openrouter_api_key: Optional[str]
    openai_api_key: Optional[str]
    anthropic_api_key: Optional[str]
    primary_provider: str = "openrouter"

@dataclass
class ProcessingConfig:
    """Document processing settings"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_length: int = 100
    max_chunks_per_doc: int = 100
    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"

@dataclass
class ToolSettings:
    """Main tool configuration"""
    environment: str
    log_level: str
    data_dir: str
    faiss_index_dir: str
    batch_size: int
    max_workers: int
    
    # Component configs
    database: DatabaseConfig
    redis: RedisConfig
    llm: LLMConfig
    processing: ProcessingConfig

def load_settings() -> ToolSettings:
    """Load settings từ environment variables"""
    
    # Parse database URL
    db_url = os.getenv('DATABASE_URL', 'postgresql://user:password@localhost:5432/database')
    # Extract components from URL for individual access
    
    return ToolSettings(
        environment=os.getenv('ENVIRONMENT', 'development'),
        log_level=os.getenv('LOG_LEVEL', 'INFO'),
        data_dir=os.getenv('DATA_DIR', './data'),
        faiss_index_dir=os.getenv('FAISS_INDEX_DIR', './data/faiss_indices'),
        batch_size=int(os.getenv('BATCH_SIZE', '10')),
        max_workers=int(os.getenv('MAX_WORKERS', '4')),
        
        database=DatabaseConfig(
            url=db_url,
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            name=os.getenv('DB_NAME', 'database'),
            user=os.getenv('DB_USER', 'user'),
            password=os.getenv('DB_PASSWORD', 'password')
        ),
        
        redis=RedisConfig(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379'))
        ),
        
        llm=LLMConfig(
            openrouter_api_key=os.getenv('OPENROUTER_API_KEY'),
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            anthropic_api_key=os.getenv('ANTHROPIC_API_KEY')
        ),
        
        processing=ProcessingConfig(
            chunk_size=int(os.getenv('CHUNK_SIZE', '512')),
            chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '50')),
            min_chunk_length=int(os.getenv('MIN_CHUNK_LENGTH', '100')),
            max_chunks_per_doc=int(os.getenv('MAX_CHUNKS_PER_DOC', '100'))
        )
    )

# Global settings instance
settings = load_settings()
```

### config/collection_config.py
```python
"""
FAISS Collection configuration và mapping rules
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

class DocumentType(Enum):
    PRODUCT_A_FEATURES = "product_a_features"
    PRODUCT_A_PRICING = "product_a_pricing"
    PRODUCT_B_FEATURES = "product_b_features"
    WARRANTY_SUPPORT = "warranty_support"
    CONTACT_COMPANY = "contact_company"

@dataclass
class CollectionConfig:
    """Configuration cho một FAISS collection"""
    name: str
    description: str
    keywords: List[str]
    doc_id_prefix: str
    expected_doc_types: List[str]
    auto_categorize_rules: Dict[str, List[str]]
    metadata_schema: Dict[str, type]
    quality_checks: List[str]

# Collection definitions
COLLECTION_CONFIGS = {
    DocumentType.PRODUCT_A_FEATURES: CollectionConfig(
        name="product_a_features",
        description="Tính năng và đặc điểm của Sản phẩm A",
        keywords=[
            "product a", "sản phẩm a", "tính năng", "feature", "chức năng",
            "khả năng", "đặc điểm", "specification", "specs", "functionality"
        ],
        doc_id_prefix="PA_FEAT",
        expected_doc_types=["pdf", "docx", "md", "txt"],
        auto_categorize_rules={
            "title_patterns": [
                r"product\s*a.*feature", r"sản phẩm\s*a.*tính năng",
                r".*feature.*product\s*a", r"chức năng.*sản phẩm\s*a"
            ],
            "content_keywords": [
                "tính năng chính", "main features", "capabilities",
                "functionality", "what can", "có thể làm"
            ],
            "section_headers": [
                "features", "tính năng", "chức năng", "khả năng",
                "đặc điểm", "specifications"
            ]
        },
        metadata_schema={
            "product": str,
            "category": str,
            "feature_type": str,
            "priority": int,
            "technical_level": str
        },
        quality_checks=[
            "min_length_check", "keyword_relevance_check", 
            "technical_accuracy_check"
        ]
    ),
    
    DocumentType.PRODUCT_A_PRICING: CollectionConfig(
        name="product_a_pricing",
        description="Giá cả và gói dịch vụ Sản phẩm A",
        keywords=[
            "product a", "sản phẩm a", "giá", "pricing", "cost", "phí",
            "gói", "plan", "subscription", "đăng ký", "thanh toán", "payment"
        ],
        doc_id_prefix="PA_PRIC",
        expected_doc_types=["pdf", "xlsx", "docx", "csv"],
        auto_categorize_rules={
            "title_patterns": [
                r"product\s*a.*pric", r"sản phẩm\s*a.*giá",
                r".*pricing.*product\s*a", r"bảng giá.*sản phẩm\s*a"
            ],
            "content_keywords": [
                "bảng giá", "price list", "pricing", "cost",
                "gói cước", "subscription", "monthly", "yearly"
            ],
            "financial_indicators": [
                "VND", "USD", "$", "đ", "triệu", "nghìn",
                "month", "year", "tháng", "năm"
            ]
        },
        metadata_schema={
            "product": str,
            "plan_type": str,
            "currency": str,
            "billing_cycle": str,
            "price_tier": str
        },
        quality_checks=[
            "price_format_check", "currency_consistency_check",
            "plan_completeness_check"
        ]
    ),
    
    DocumentType.WARRANTY_SUPPORT: CollectionConfig(
        name="warranty_support",
        description="Thông tin bảo hành và hỗ trợ khách hàng",
        keywords=[
            "bảo hành", "warranty", "guarantee", "đảm bảo", "chính sách",
            "hỗ trợ", "support", "khách hàng", "customer", "service"
        ],
        doc_id_prefix="WARR_SUPP",
        expected_doc_types=["pdf", "docx", "md"],
        auto_categorize_rules={
            "title_patterns": [
                r"bảo hành", r"warranty", r"support", r"hỗ trợ",
                r"customer.*service", r"dịch vụ.*khách hàng"
            ],
            "content_keywords": [
                "chính sách bảo hành", "warranty policy", "support",
                "customer service", "technical support", "hỗ trợ kỹ thuật"
            ],
            "policy_indicators": [
                "policy", "chính sách", "terms", "điều khoản",
                "conditions", "điều kiện"
            ]
        },
        metadata_schema={
            "policy_type": str,
            "warranty_period": str,
            "coverage_type": str,
            "support_level": str
        },
        quality_checks=[
            "policy_completeness_check", "date_validity_check",
            "legal_compliance_check"
        ]
    )
    # ... other collections
}

def get_collection_by_keywords(text: str) -> Optional[DocumentType]:
    """Auto-detect collection dựa trên keywords trong text"""
    text_lower = text.lower()
    
    scores = {}
    for doc_type, config in COLLECTION_CONFIGS.items():
        score = 0
        for keyword in config.keywords:
            if keyword.lower() in text_lower:
                score += 1
        
        # Bonus for title patterns
        for pattern in config.auto_categorize_rules.get("title_patterns", []):
            import re
            if re.search(pattern, text_lower):
                score += 3
        
        if score > 0:
            scores[doc_type] = score
    
    return max(scores.items(), key=lambda x: x[1])[0] if scores else None
```

## 4. Core Components

### core/document_parser.py
```python
"""
Multi-format document parser
"""
import os
import re
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass
import logging

# Document processing imports
import PyPDF2
from docx import Document as DocxDocument
import pandas as pd
from bs4 import BeautifulSoup
import markdown

logger = logging.getLogger(__name__)

@dataclass
class ParsedDocument:
    """Container cho parsed document data"""
    filename: str
    file_type: str
    title: str
    content: str
    raw_metadata: Dict[str, Any]
    page_count: Optional[int] = None
    word_count: int = 0
    
    def __post_init__(self):
        self.word_count = len(self.content.split())

class DocumentParser:
    """Multi-format document parser"""
    
    SUPPORTED_FORMATS = {
        '.pdf': 'parse_pdf',
        '.docx': 'parse_docx',
        '.txt': 'parse_txt',
        '.md': 'parse_markdown',
        '.csv': 'parse_csv',
        '.xlsx': 'parse_excel',
        '.html': 'parse_html'
    }
    
    def __init__(self):
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'by_format': {}
        }
    
    def parse_file(self, file_path: Path) -> Optional[ParsedDocument]:
        """Parse single file based on extension"""
        
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        file_ext = file_path.suffix.lower()
        if file_ext not in self.SUPPORTED_FORMATS:
            logger.warning(f"Unsupported format: {file_ext}")
            return None
        
        parser_method = getattr(self, self.SUPPORTED_FORMATS[file_ext])
        
        try:
            self.stats['total_processed'] += 1
            result = parser_method(file_path)
            
            if result:
                self.stats['successful'] += 1
                self.stats['by_format'][file_ext] = self.stats['by_format'].get(file_ext, 0) + 1
                logger.info(f"Successfully parsed: {file_path.name}")
            else:
                self.stats['failed'] += 1
                logger.error(f"Failed to parse: {file_path.name}")
            
            return result
            
        except Exception as e:
            self.stats['failed'] += 1
            logger.error(f"Error parsing {file_path.name}: {str(e)}")
            return None
    
    def parse_pdf(self, file_path: Path) -> Optional[ParsedDocument]:
        """Parse PDF file"""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                metadata = {}
                if pdf_reader.metadata:
                    metadata = {
                        'title': pdf_reader.metadata.get('/Title', ''),
                        'author': pdf_reader.metadata.get('/Author', ''),
                        'subject': pdf_reader.metadata.get('/Subject', ''),
                        'creator': pdf_reader.metadata.get('/Creator', '')
                    }
                
                # Extract text từ all pages
                content_parts = []
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text = page.extract_text()
                        if text.strip():
                            content_parts.append(f"--- Page {page_num + 1} ---\n{text}")
                    except Exception as e:
                        logger.warning(f"Error extracting page {page_num + 1}: {e}")
                
                content = '\n\n'.join(content_parts)
                title = metadata.get('title') or self._extract_title_from_content(content) or file_path.stem
                
                return ParsedDocument(
                    filename=file_path.name,
                    file_type='pdf',
                    title=title,
                    content=content,
                    raw_metadata=metadata,
                    page_count=len(pdf_reader.pages)
                )
                
        except Exception as e:
            logger.error(f"PDF parsing error for {file_path.name}: {e}")
            return None
    
    def parse_docx(self, file_path: Path) -> Optional[ParsedDocument]:
        """Parse DOCX file"""
        try:
            doc = DocxDocument(file_path)
            
            # Extract content
            content_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    content_parts.append(paragraph.text)
            
            content = '\n'.join(content_parts)
            
            # Extract metadata từ document properties
            props = doc.core_properties
            metadata = {
                'title': props.title or '',
                'author': props.author or '',
                'subject': props.subject or '',
                'created': str(props.created) if props.created else '',
                'modified': str(props.modified) if props.modified else ''
            }
            
            title = metadata.get('title') or self._extract_title_from_content(content) or file_path.stem
            
            return ParsedDocument(
                filename=file_path.name,
                file_type='docx',
                title=title,
                content=content,
                raw_metadata=metadata,
                page_count=None  # DOCX không có page concept rõ ràng
            )
            
        except Exception as e:
            logger.error(f"DOCX parsing error for {file_path.name}: {e}")
            return None
    
    def parse_txt(self, file_path: Path) -> Optional[ParsedDocument]:
        """Parse text file"""
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                logger.error(f"Could not decode text file: {file_path.name}")
                return None
            
            title = self._extract_title_from_content(content) or file_path.stem
            
            return ParsedDocument(
                filename=file_path.name,
                file_type='txt',
                title=title,
                content=content,
                raw_metadata={'encoding': encoding},
                page_count=None
            )
            
        except Exception as e:
            logger.error(f"TXT parsing error for {file_path.name}: {e}")
            return None
    
    def parse_markdown(self, file_path: Path) -> Optional[ParsedDocument]:
        """Parse Markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
            
            # Convert to HTML để extract clean text
            html = markdown.markdown(md_content)
            soup = BeautifulSoup(html, 'html.parser')
            content = soup.get_text()
            
            # Extract title từ first # header hoặc filename
            title_match = re.search(r'^#\s+(.+)$', md_content, re.MULTILINE)
            title = title_match.group(1) if title_match else file_path.stem
            
            return ParsedDocument(
                filename=file_path.name,
                file_type='md',
                title=title,
                content=content,
                raw_metadata={'original_markdown': md_content[:500]},  # First 500 chars
                page_count=None
            )
            
        except Exception as e:
            logger.error(f"Markdown parsing error for {file_path.name}: {e}")
            return None
    
    def _extract_title_from_content(self, content: str) -> Optional[str]:
        """Extract title từ document content"""
        lines = content.strip().split('\n')
        
        # Look for first non-empty line as potential title
        for line in lines[:5]:  # Check first 5 lines
            line = line.strip()
            if line and len(line) < 100:  # Reasonable title length
                return line
        
        return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parsing statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset parsing statistics"""
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'by_format': {}
        }
```

## 5. CLI Interface

### main.py
```python
"""
CLI interface cho Data Ingestion Tool
"""
import click
import asyncio
import logging
from pathlib import Path
from typing import Optional

from config.settings import settings
from core.document_parser import DocumentParser
from core.content_processor import ContentProcessor
from core.collection_manager import CollectionManager
from core.metadata_manager import MetadataManager
from utils.progress_tracker import ProgressTracker

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """Enterprise Chatbot - Data Ingestion Tool"""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

@cli.command()
@click.argument('input_path', type=click.Path(exists=True))
@click.option('--collection', '-c', 
              type=click.Choice(['product_a_features', 'product_a_pricing', 
                               'product_b_features', 'warranty_support', 
                               'contact_company', 'auto']), 
              default='auto',
              help='Target collection (auto = auto-detect)')
@click.option('--batch-size', '-b', default=10, help='Batch processing size')
@click.option('--dry-run', is_flag=True, help='Simulate processing without saving')
def ingest(input_path: str, collection: str, batch_size: int, dry_run: bool):
    """Ingest documents into FAISS collections"""
    
    input_path = Path(input_path)
    
    if input_path.is_file():
        files = [input_path]
    else:
        # Recursively find supported files
        supported_extensions = {'.pdf', '.docx', '.txt', '.md', '.csv', '.xlsx'}
        files = [f for f in input_path.rglob('*') 
                if f.is_file() and f.suffix.lower() in supported_extensions]
    
    if not files:
        click.echo("No supported files found!")
        return
    
    click.echo(f"Found {len(files)} files to process")
    
    if dry_run:
        click.echo("DRY RUN MODE - No changes will be saved")
    
    # Run async processing
    asyncio.run(process_files(files, collection, batch_size, dry_run))

async def process_files(files: list, target_collection: str, 
                       batch_size: int, dry_run: bool):
    """Async file processing"""
    
    # Initialize components
    parser = DocumentParser()
    processor = ContentProcessor()
    collection_manager = CollectionManager()
    metadata_manager = MetadataManager()
    progress = ProgressTracker(total=len(files))
    
    try:
        # Initialize connections
        await collection_manager.initialize()
        await metadata_manager.initialize()
        
        successful = 0
        failed = 0
        
        # Process in batches
        for i in range(0, len(files), batch_size):
            batch = files[i:i + batch_size]
            
            for file_path in batch:
                try:
                    # Parse document
                    parsed_doc = parser.parse_file(file_path)
                    if not parsed_doc:
                        failed += 1
                        progress.update(1, f"Failed to parse: {file_path.name}")
                        continue
                    
                    # Process content
                    processed_data = await processor.process_document(
                        parsed_doc, target_collection
                    )
                    
                    if not processed_data:
                        failed += 1
                        progress.update(1, f"Failed to process: {file_path.name}")
                        continue
                    
                    # Save to storage (unless dry run)
                    if not dry_run:
                        # Add to FAISS collection
                        await collection_manager.add_documents(
                            processed_data['collection'],
                            processed_data['chunks']
                        )
                        
                        # Save metadata
                        await metadata_manager.save_document_metadata(
                            processed_data['document_metadata']
                        )
                    
                    successful += 1
                    progress.update(1, f"Processed: {file_path.name}")
                    
                except Exception as e:
                    failed += 1
                    logger.error(f"Error processing {file_path}: {e}")
                    progress.update(1, f"Error: {file_path.name}")
        
        # Final summary
        click.echo(f"\nProcessing complete!")
        click.echo(f"Successful: {successful}")
        click.echo(f"Failed: {failed}")
        
        # Show parsing stats
        stats = parser.get_stats()
        click.echo(f"\nParsing Statistics:")
        click.echo(f"Total files: {stats['total_processed']}")
        click.echo(f"By format: {stats['by_format']}")
        
    finally:
        await collection_manager.cleanup()
        await metadata_manager.cleanup()

@cli.command()
def status():
    """Show collection status and statistics"""
    
    async def show_status():
        collection_manager = CollectionManager()
        metadata_manager = MetadataManager()
        
        try:
            await collection_manager.initialize()
            await metadata_manager.initialize()
            
            # Collection statistics
            click.echo("=== FAISS Collections Status ===")
            collections = await collection_manager.get_all_collections_info()
            
            for name, info in collections.items():
                click.echo(f"\n{name}:")
                click.echo(f"  Documents: {info['doc_count']}")
                click.echo(f"  Index size: {info['index_size']}")
                click.echo(f"  Last updated: {info['last_updated']}")
            
            # Database statistics
            click.echo("\n=== Database Status ===")
            db_stats = await metadata_manager.get_statistics()
            click.echo(f"Total documents: {db_stats['total_documents']}")
            click.echo(f"Total chunks: {db_stats['total_chunks']}")
            click.echo(f"By collection: {db_stats['by_collection']}")
            
        finally:
            await collection_manager.cleanup()
            await metadata_manager.cleanup()
    
    asyncio.run(show_status())

@cli.command()
@click.option('--collection', '-c', help='Collection name to validate')
@click.option('--check-embeddings', is_flag=True, help='Validate embedding quality')
def validate(collection: Optional[str], check_embeddings: bool):
    """Validate collection data quality"""
    
    async def run_validation():
        from analyzers.quality_checker import QualityChecker
        
        checker = QualityChecker()
        await checker.initialize()
        
        try:
            if collection:
                results = await checker.validate_collection(collection)
                click.echo(f"Validation results for {collection}:")
            else:
                results = await checker.validate_all_collections()
                click.echo("Validation results for all collections:")
            
            for coll_name, report in results.items():
                click.echo(f"\n{coll_name}:")
                click.echo(f"  Quality score: {report['quality_score']:.2f}")
                click.echo(f"  Issues found: {len(report['issues'])}")
                
                if report['issues']:
                    for issue in report['issues'][:5]:  # Show first 5 issues
                        click.echo(f"    - {issue}")
        
        finally:
            await checker.cleanup()
    
    asyncio.run(run_validation())

if __name__ == '__main__':
    cli()
```

## 6. Web Interface

### web_interface.py
```python
"""
Streamlit-based web interface cho data ingestion
"""
import streamlit as st
import asyncio
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from config.settings import settings
from core.document_parser import DocumentParser
from core.collection_manager import CollectionManager
from analyzers.quality_checker import QualityChecker

st.set_page_config(
    page_title="Data Ingestion Tool",
    page_icon="📊",
    layout="wide"
)

def main():
    st.title("📊 Enterprise Chatbot - Data Ingestion Tool")
    st.sidebar.title("Navigation")
    
    pages = {
        "Document Upload": page_upload,
        "Collection Status": page_status,
        "Quality Check": page_quality,
        "Settings": page_settings
    }
    
    selected_page = st.sidebar.selectbox("Select Page", list(pages.keys
	
## 6. Web Interface (tiếp tục)

### web_interface.py (tiếp tục)
```python
    selected_page = st.sidebar.selectbox("Select Page", list(pages.keys()))
    pages[selected_page]()

def page_upload():
    """Document upload and processing page"""
    st.header("Document Upload & Processing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File upload
        uploaded_files = st.file_uploader(
            "Upload documents",
            type=['pdf', 'docx', 'txt', 'md', 'csv', 'xlsx'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT, MD, CSV, XLSX"
        )
        
        # Collection selection
        collection_options = [
            'auto',
            'product_a_features',
            'product_a_pricing', 
            'product_b_features',
            'warranty_support',
            'contact_company'
        ]
        
        target_collection = st.selectbox(
            "Target Collection",
            collection_options,
            help="Select 'auto' for automatic categorization"
        )
        
        # Processing options
        with st.expander("Advanced Options"):
            batch_size = st.slider("Batch Size", 1, 20, 5)
            dry_run = st.checkbox("Dry Run (preview only)", value=False)
            auto_keywords = st.checkbox("Auto-extract keywords", value=True)
            quality_check = st.checkbox("Enable quality validation", value=True)
    
    with col2:
        st.subheader("Processing Settings")
        st.write(f"Environment: {settings.environment}")
        st.write(f"Batch Size: {batch_size}")
        st.write(f"Target Collection: {target_collection}")
        
        if dry_run:
            st.warning("Dry run mode - no changes will be saved")
    
    # Process button
    if st.button("Process Documents", type="primary", disabled=not uploaded_files):
        if uploaded_files:
            process_uploaded_files(uploaded_files, target_collection, batch_size, dry_run)

def process_uploaded_files(uploaded_files, target_collection, batch_size, dry_run):
    """Process uploaded files"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_container = st.empty()
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Save uploaded files
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = temp_path / uploaded_file.name
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            file_paths.append(file_path)
        
        # Run processing
        results = asyncio.run(process_files_async(
            file_paths, target_collection, batch_size, dry_run,
            progress_bar, status_text
        ))
        
        # Display results
        display_processing_results(results, results_container)

async def process_files_async(file_paths, target_collection, batch_size, dry_run, 
                            progress_bar, status_text):
    """Async file processing with progress updates"""
    
    parser = DocumentParser()
    results = {
        'successful': 0,
        'failed': 0,
        'details': [],
        'parsing_stats': {},
        'errors': []
    }
    
    total_files = len(file_paths)
    
    for i, file_path in enumerate(file_paths):
        try:
            status_text.text(f"Processing: {file_path.name}")
            
            # Parse document
            parsed_doc = parser.parse_file(file_path)
            
            if parsed_doc:
                results['successful'] += 1
                results['details'].append({
                    'filename': file_path.name,
                    'status': 'success',
                    'title': parsed_doc.title,
                    'word_count': parsed_doc.word_count,
                    'file_type': parsed_doc.file_type,
                    'detected_collection': target_collection
                })
            else:
                results['failed'] += 1
                results['details'].append({
                    'filename': file_path.name,
                    'status': 'failed',
                    'error': 'Parse failed'
                })
                results['errors'].append(f"Failed to parse: {file_path.name}")
            
            # Update progress
            progress = (i + 1) / total_files
            progress_bar.progress(progress)
            
        except Exception as e:
            results['failed'] += 1
            results['errors'].append(f"Error processing {file_path.name}: {str(e)}")
    
    results['parsing_stats'] = parser.get_stats()
    status_text.text("Processing complete!")
    
    return results

def display_processing_results(results, container):
    """Display processing results in structured format"""
    
    with container.container():
        st.subheader("Processing Results")
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Successful", results['successful'])
        with col2:
            st.metric("Failed", results['failed'])
        with col3:
            total = results['successful'] + results['failed']
            success_rate = (results['successful'] / total * 100) if total > 0 else 0
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Detailed results table
        if results['details']:
            df = pd.DataFrame(results['details'])
            st.dataframe(df, use_container_width=True)
        
        # Parsing statistics
        if results['parsing_stats']:
            with st.expander("Parsing Statistics"):
                stats = results['parsing_stats']
                st.json(stats)
        
        # Errors
        if results['errors']:
            with st.expander("Errors", expanded=True):
                for error in results['errors']:
                    st.error(error)

def page_status():
    """Collection status and statistics page"""
    st.header("Collection Status")
    
    # Refresh button
    if st.button("Refresh Status"):
        st.rerun()
    
    # Load collection status
    status_data = asyncio.run(load_collection_status())
    
    if status_data:
        # Overview metrics
        st.subheader("Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        total_docs = sum(info['doc_count'] for info in status_data['collections'].values())
        total_collections = len(status_data['collections'])
        
        with col1:
            st.metric("Total Collections", total_collections)
        with col2:
            st.metric("Total Documents", total_docs)
        with col3:
            avg_docs = total_docs / total_collections if total_collections > 0 else 0
            st.metric("Avg Docs/Collection", f"{avg_docs:.1f}")
        with col4:
            st.metric("System Status", status_data['overall_status'])
        
        # Collection details
        st.subheader("Collection Details")
        
        for collection_name, info in status_data['collections'].items():
            with st.expander(f"{collection_name} ({info['doc_count']} documents)"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Document Count:** {info['doc_count']}")
                    st.write(f"**Index Size:** {info['index_size']}")
                
                with col2:
                    st.write(f"**Last Updated:** {info.get('last_updated', 'Unknown')}")
                    st.write(f"**Status:** {info.get('status', 'Unknown')}")
                
                with col3:
                    if st.button(f"View Details", key=f"details_{collection_name}"):
                        show_collection_details(collection_name)

async def load_collection_status():
    """Load collection status data"""
    try:
        collection_manager = CollectionManager()
        await collection_manager.initialize()
        
        collections_info = await collection_manager.get_all_collections_info()
        
        # Determine overall status
        all_healthy = all(info.get('status') == 'healthy' for info in collections_info.values())
        overall_status = 'Healthy' if all_healthy else 'Needs Attention'
        
        await collection_manager.cleanup()
        
        return {
            'collections': collections_info,
            'overall_status': overall_status
        }
    except Exception as e:
        st.error(f"Error loading collection status: {e}")
        return None

def page_quality():
    """Quality check and validation page"""
    st.header("Data Quality Check")
    
    # Quality check options
    col1, col2 = st.columns([1, 1])
    
    with col1:
        check_type = st.radio(
            "Check Type",
            ["All Collections", "Specific Collection", "Recent Uploads"]
        )
        
        if check_type == "Specific Collection":
            collection_name = st.selectbox(
                "Select Collection",
                ['product_a_features', 'product_a_pricing', 'product_b_features', 
                 'warranty_support', 'contact_company']
            )
    
    with col2:
        quality_checks = st.multiselect(
            "Quality Checks to Run",
            ["Content Length", "Keyword Relevance", "Duplicate Detection", 
             "Embedding Quality", "Metadata Completeness"],
            default=["Content Length", "Keyword Relevance"]
        )
    
    # Run quality check
    if st.button("Run Quality Check", type="primary"):
        run_quality_check(check_type, quality_checks, 
                         collection_name if check_type == "Specific Collection" else None)

def run_quality_check(check_type, quality_checks, collection_name=None):
    """Execute quality check"""
    
    progress_container = st.empty()
    results_container = st.empty()
    
    with progress_container:
        with st.spinner("Running quality checks..."):
            results = asyncio.run(execute_quality_check_async(
                check_type, quality_checks, collection_name
            ))
    
    progress_container.empty()
    display_quality_results(results, results_container)

async def execute_quality_check_async(check_type, quality_checks, collection_name):
    """Execute quality check asynchronously"""
    
    from analyzers.quality_checker import QualityChecker
    
    checker = QualityChecker()
    await checker.initialize()
    
    try:
        if check_type == "All Collections":
            results = await checker.validate_all_collections()
        elif check_type == "Specific Collection" and collection_name:
            results = await checker.validate_collection(collection_name)
        else:
            results = {}
        
        return results
    
    finally:
        await checker.cleanup()

def display_quality_results(results, container):
    """Display quality check results"""
    
    with container:
        st.subheader("Quality Check Results")
        
        if not results:
            st.warning("No results to display")
            return
        
        for collection_name, report in results.items():
            with st.expander(f"{collection_name} - Score: {report.get('quality_score', 0):.2f}"):
                
                # Quality score gauge
                score = report.get('quality_score', 0)
                color = "green" if score >= 0.8 else "orange" if score >= 0.6 else "red"
                st.markdown(f"**Quality Score:** :{color}[{score:.2f}]")
                
                # Issues summary
                issues = report.get('issues', [])
                if issues:
                    st.write(f"**Issues Found:** {len(issues)}")
                    for issue in issues[:10]:  # Show first 10 issues
                        st.write(f"• {issue}")
                    
                    if len(issues) > 10:
                        st.write(f"... and {len(issues) - 10} more issues")
                else:
                    st.success("No issues found!")
                
                # Recommendations
                recommendations = report.get('recommendations', [])
                if recommendations:
                    st.write("**Recommendations:**")
                    for rec in recommendations:
                        st.info(rec)

def page_settings():
    """Settings and configuration page"""
    st.header("Settings & Configuration")
    
    # Environment info
    st.subheader("Environment Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Environment:** {settings.environment}")
        st.write(f"**Log Level:** {settings.log_level}")
        st.write(f"**Data Directory:** {settings.data_dir}")
        st.write(f"**FAISS Index Directory:** {settings.faiss_index_dir}")
    
    with col2:
        st.write(f"**Batch Size:** {settings.batch_size}")
        st.write(f"**Max Workers:** {settings.max_workers}")
        st.write(f"**Redis Host:** {settings.redis.host}")
        st.write(f"**Database URL:** {settings.database.url}")
    
    # Processing configuration
    st.subheader("Processing Configuration")
    with st.expander("Chunk Settings"):
        st.write(f"**Chunk Size:** {settings.processing.chunk_size}")
        st.write(f"**Chunk Overlap:** {settings.processing.chunk_overlap}")
        st.write(f"**Min Chunk Length:** {settings.processing.min_chunk_length}")
        st.write(f"**Max Chunks per Document:** {settings.processing.max_chunks_per_doc}")
        st.write(f"**Embedding Model:** {settings.processing.embedding_model}")
    
    # API Keys status
    st.subheader("API Keys Status")
    llm_config = settings.llm
    
    api_status = []
    if llm_config.openrouter_api_key:
        api_status.append(("OpenRouter", "✅ Configured"))
    else:
        api_status.append(("OpenRouter", "❌ Not configured"))
    
    if llm_config.openai_api_key:
        api_status.append(("OpenAI", "✅ Configured"))
    else:
        api_status.append(("OpenAI", "❌ Not configured"))
    
    if llm_config.anthropic_api_key:
        api_status.append(("Anthropic", "✅ Configured"))
    else:
        api_status.append(("Anthropic", "❌ Not configured"))
    
    df_apis = pd.DataFrame(api_status, columns=["Provider", "Status"])
    st.dataframe(df_apis, use_container_width=True)
    
    # Collection configurations
    st.subheader("Collection Configurations")
    from config.collection_config import COLLECTION_CONFIGS
    
    for doc_type, config in COLLECTION_CONFIGS.items():
        with st.expander(f"{config.name}"):
            st.write(f"**Description:** {config.description}")
            st.write(f"**Keywords:** {', '.join(config.keywords[:5])}...")
            st.write(f"**Doc ID Prefix:** {config.doc_id_prefix}")
            st.write(f"**Expected Types:** {', '.join(config.expected_doc_types)}")

if __name__ == "__main__":
    main()
```

## 7. Remaining Core Components

### core/content_processor.py
```python
"""
Content processing và chunking engine
"""
import re
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
import logging

from sentence_transformers import SentenceTransformer
import numpy as np

from config.settings import settings
from config.collection_config import get_collection_by_keywords, COLLECTION_CONFIGS
from core.document_parser import ParsedDocument

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Single document chunk với metadata"""
    chunk_id: str
    content: str
    embedding: Optional[np.ndarray]
    metadata: Dict[str, Any]
    chunk_index: int
    word_count: int

@dataclass
class ProcessedDocument:
    """Complete processed document data"""
    document_id: str
    collection: str
    chunks: List[DocumentChunk]
    document_metadata: Dict[str, Any]
    processing_stats: Dict[str, Any]

class ContentProcessor:
    """Process và chunk documents cho FAISS storage"""
    
    def __init__(self):
        self.embedding_model = None
        self.chunk_size = settings.processing.chunk_size
        self.chunk_overlap = settings.processing.chunk_overlap
        self.min_chunk_length = settings.processing.min_chunk_length
        self.max_chunks_per_doc = settings.processing.max_chunks_per_doc
        
    async def initialize(self):
        """Initialize embedding model"""
        if not self.embedding_model:
            model_name = settings.processing.embedding_model
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Loaded embedding model: {model_name}")
    
    async def process_document(self, parsed_doc: ParsedDocument, 
                             target_collection: str = "auto") -> Optional[ProcessedDocument]:
        """Process single document through complete pipeline"""
        
        if not self.embedding_model:
            await self.initialize()
        
        try:
            # Determine target collection
            if target_collection == "auto":
                detected_collection = self._auto_detect_collection(parsed_doc)
                if not detected_collection:
                    logger.warning(f"Could not auto-detect collection for {parsed_doc.filename}")
                    return None
                collection_name = detected_collection.value
            else:
                collection_name = target_collection
            
            # Generate document ID
            doc_id = self._generate_document_id(parsed_doc, collection_name)
            
            # Extract and clean content
            cleaned_content = self._clean_content(parsed_doc.content)
            
            # Create chunks
            text_chunks = self._create_text_chunks(cleaned_content)
            
            if not text_chunks:
                logger.warning(f"No valid chunks created for {parsed_doc.filename}")
                return None
            
            # Create document chunks với embeddings
            chunks = []
            for i, chunk_text in enumerate(text_chunks):
                if len(chunk_text.strip()) < self.min_chunk_length:
                    continue
                
                # Generate embedding
                embedding = await self._generate_embedding(chunk_text)
                
                # Create chunk metadata
                chunk_metadata = {
                    'document_id': doc_id,
                    'chunk_index': i,
                    'collection': collection_name,
                    'source_file': parsed_doc.filename,
                    'file_type': parsed_doc.file_type,
                    'title': parsed_doc.title,
                    'created_at': asyncio.get_event_loop().time()
                }
                
                # Add collection-specific metadata
                chunk_metadata.update(
                    self._extract_collection_metadata(chunk_text, collection_name)
                )
                
                chunk_id = f"{doc_id}_chunk_{i:03d}"
                
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=chunk_text,
                    embedding=embedding,
                    metadata=chunk_metadata,
                    chunk_index=i,
                    word_count=len(chunk_text.split())
                )
                
                chunks.append(chunk)
            
            if not chunks:
                logger.warning(f"No valid chunks with embeddings for {parsed_doc.filename}")
                return None
            
            # Create document metadata
            document_metadata = {
                'document_id': doc_id,
                'filename': parsed_doc.filename,
                'title': parsed_doc.title,
                'collection': collection_name,
                'file_type': parsed_doc.file_type,
                'original_content_length': len(parsed_doc.content),
                'chunk_count': len(chunks),
                'total_word_count': parsed_doc.word_count,
                'processing_timestamp': asyncio.get_event_loop().time(),
                'raw_metadata': parsed_doc.raw_metadata
            }
            
            # Add extracted keywords
            keywords = await self._extract_keywords(cleaned_content, collection_name)
            document_metadata['extracted_keywords'] = keywords
            
            processing_stats = {
                'original_length': len(parsed_doc.content),
                'cleaned_length': len(cleaned_content),
                'chunks_created': len(chunks),
                'avg_chunk_size': np.mean([len(c.content) for c in chunks]),
                'embedding_dimension': len(chunks[0].embedding) if chunks else 0
            }
            
            return ProcessedDocument(
                document_id=doc_id,
                collection=collection_name,
                chunks=chunks,
                document_metadata=document_metadata,
                processing_stats=processing_stats
            )
            
        except Exception as e:
            logger.error(f"Error processing document {parsed_doc.filename}: {e}")
            return None
    
    def _auto_detect_collection(self, parsed_doc: ParsedDocument) -> Optional[str]:
        """Auto-detect target collection based on content"""
        
        # Combine title và content sample cho detection
        detection_text = f"{parsed_doc.title} {parsed_doc.content[:1000]}"
        
        return get_collection_by_keywords(detection_text)
    
    def _generate_document_id(self, parsed_doc: ParsedDocument, collection: str) -> str:
        """Generate unique document ID"""
        
        # Get collection config
        config = None
        for doc_type, conf in COLLECTION_CONFIGS.items():
            if conf.name == collection:
                config = conf
                break
        
        prefix = config.doc_id_prefix if config else "DOC"
        
        # Create hash từ filename và content
        content_hash = hashlib.md5(
            f"{parsed_doc.filename}{parsed_doc.content[:500]}".encode()
        ).hexdigest()[:8]
        
        return f"{prefix}_{content_hash.upper()}"
    
    def _clean_content(self, content: str) -> str:
        """Clean và normalize content"""
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove page numbers và headers/footers patterns
        content = re.sub(r'Page \d+', '', content)
        content = re.sub(r'\d+/\d+', '', content)
        
        # Remove special characters but keep Vietnamese
        content = re.sub(r'[^\w\sÀ-ỹ.,!?;:()\-"\'\/]', ' ', content)
        
        # Normalize punctuation spacing
        content = re.sub(r'\s*([.,!?;:])\s*', r'\1 ', content)
        
        return content.strip()
    
    def _create_text_chunks(self, content: str) -> List[str]:
        """Create overlapping text chunks"""
        
        words = content.split()
        
        if len(words) <= self.chunk_size:
            return [content]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            if len(chunk_text.strip()) >= self.min_chunk_length:
                chunks.append(chunk_text)
            
            if end >= len(words):
                break
                
            start += self.chunk_size - self.chunk_overlap
            
            # Safety check
            if len(chunks) >= self.max_chunks_per_doc:
                logger.warning(f"Reached max chunks limit: {self.max_chunks_per_doc}")
                break
        
        return chunks
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text chunk"""
        
        try:
            # SentenceTransformers encode
            embedding = self.embedding_model.encode([text])[0]
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return np.zeros(self.embedding_model.get_sentence_embedding_dimension(), 
                          dtype=np.float32)
    
    def _extract_collection_metadata(self, chunk_text: str, collection: str) -> Dict[str, Any]:
        """Extract collection-specific metadata"""
        
        metadata = {}
        
        if collection == "product_a_features" or collection == "product_b_features":
            # Extract feature-related metadata
            if any(word in chunk_text.lower() for word in ["security", "bảo mật"]):
                metadata["feature_type"] = "security"
            elif any(word in chunk_text.lower() for word in ["integration", "tích hợp"]):
                metadata["feature_type"] = "integration"
            elif any(word in chunk_text.lower() for word in ["analytics", "phân tích"]):
                metadata["feature_type"] = "analytics"
            else:
                metadata["feature_type"] = "general"
                
            # Technical level detection
            technical_words = ["API", "SDK", "algorithm", "database", "server"]
            if any(word in chunk_text for word in technical_words):
                metadata["technical_level"] = "advanced"
            else:
                metadata["technical_level"] = "basic"
                
        elif collection == "product_a_pricing" or collection == "product_b_pricing":
            # Extract pricing-related metadata
            if any(word in chunk_text.lower() for word in ["monthly", "hàng tháng"]):
                metadata["billing_cycle"] = "monthly"
            elif any(word in chunk_text.lower() for word in ["yearly", "hàng năm"]):
                metadata["billing_cycle"] = "yearly"
                
            # Price tier detection
            if any(word in chunk_text.lower() for word in ["basic", "cơ bản"]):
                metadata["price_tier"] = "basic"
            elif any(word in chunk_text.lower() for word in ["premium", "cao cấp"]):
                metadata["price_tier"] = "premium"
            elif any(word in chunk_text.lower() for word in ["enterprise", "doanh nghiệp"]):
                metadata["price_tier"] = "enterprise"
        
        return metadata
    
    async def _extract_keywords(self, content: str, collection: str) -> List[str]:
        """Extract relevant keywords từ content"""
        
        # Simple keyword extraction using frequency và relevance
        words = re.findall(r'\w+', content.lower())
        
        # Filter out common stop words
        stop_words = {
            'và', 'của', 'trong', 'với', 'để', 'là', 'có', 'được', 'một', 'các',
            'and', 'the', 'in', 'to', 'of', 'a', 'is', 'are', 'was', 'were'
        }
        
        # Count word frequencies
        word_freq = {}
        for word in words:
            if len(word) > 2 and word not in stop_words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get collection-specific keywords
        config = None
        for doc_type, conf in COLLECTION_CONFIGS.items():
            if conf.name == collection:
                config = conf
                break
        
        relevant_keywords = []
        if config:
            # Prioritize collection-specific keywords
            for keyword in config.keywords:
                if keyword.lower() in content.lower():
                    relevant_keywords.append(keyword)
        
        # Add high-frequency words
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        for word, freq in top_words:
            if word not in relevant_keywords and freq > 2:
                relevant_keywords.append(word)
        
        return relevant_keywords[:15]  # Limit to 15 keywords
```

## 8. Additional Environment Variables

Thêm vào file .env:

```bash
# Data Ingestion Tool Settings
DATA_DIR=./data
FAISS_INDEX_DIR=./data/faiss_indices
BATCH_SIZE=10
MAX_WORKERS=4

# Processing Settings
CHUNK_SIZE=512
CHUNK_OVERLAP=50
MIN_CHUNK_LENGTH=100
MAX_CHUNKS_PER_DOC=100
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2

# Quality Check Settings
QUALITY_CHECK_ENABLED=true
MIN_QUALITY_SCORE=0.7
AUTO_CATEGORIZATION_THRESHOLD=0.8
```

## 9. Usage Guide

### CLI Usage:
```bash
# Install dependencies
pip install -r requirements.txt

# Ingest single file
python main.py ingest /path/to/document.pdf --collection auto

# Ingest directory (batch processing)
python main.py ingest /path/to/docs/ --batch-size 5 --collection product_a_features

# Check collection status
python main.py status

# Validate data quality
python main.py validate --collection product_a_features --check-embeddings

# Dry run (preview only)
python main.py ingest /path/to/docs/ --dry-run
```

### Web Interface:
```bash
# Start web interface
streamlit run web_interface.py --server.port 8501
```

### Key Features:
1. **Multi-format Support**: PDF, DOCX, TXT, MD, CSV, XLSX
2. **Auto-categorization**: Intelligent collection assignment
3. **Quality Validation**: Content quality checks
4. **Batch Processing**: Efficient bulk operations
5. **Progress Tracking**: Real-time processing status
6. **Metadata Extraction**: Rich document metadata
7. **Embedding Generation**: High-quality vector representations
8. **Error Handling**: Robust error recovery
9. **Dry Run Mode**: Preview without changes
10. **Web Interface**: User-friendly GUI option

Tool này tích hợp hoàn toàn với hệ thống chatbot hiện tại và sử dụng cùng configuration từ .env file.	