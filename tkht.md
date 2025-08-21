Tôi sẽ giúp bạn xây dựng tool quản lý database theo thiết kế đã phác thảo. Dựa trên tài liệu hiện có, tôi sẽ tạo một hệ thống data ingestion hoàn chỉnh với hỗ trợ tiếng Việt.

## Cấu trúc dự án

Trước tiên, hãy tạo cấu trúc thư mục cho tool data ingestion:

```bash
mkdir data_ingestion_tool
cd data_ingestion_tool
mkdir -p config core models utils web tests/sample_documents
```

## 1. Requirements và Dependencies

```python
# requirements.txt
fastapi==0.104.1
uvicorn==0.24.0
pydantic==2.5.0
python-dotenv==1.0.0
psycopg2-binary==2.9.7
redis==5.0.1
sentence-transformers==2.2.2
faiss-cpu==1.7.4
numpy==1.24.3
pandas==2.1.3
PyPDF2==3.0.1
python-docx==0.8.11
markdown==3.5.1
beautifulsoup4==4.12.2
pyvi==0.1.1  # Thư viện xử lý tiếng Việt
pyyaml==6.0.1
click==8.1.7
tqdm==4.66.1
langdetect==1.0.9
scikit-learn==1.3.2
```

## 2. Configuration Files

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
      - "đặc điểm"
    categories:
      - security
      - performance
      - integration
      - ui_ux
      - api
    max_chunk_size: 500
    overlap_size: 50
    embedding_model: "keepitreal/vietnamese-sbert"
    
  product_a_pricing:
    prefix: "PA_PRIC"  
    description: "Product A Pricing and Plans"
    keywords_mapping:
      - "giá cả"
      - "pricing" 
      - "gói dịch vụ"
      - "chi phí"
      - "bảng giá"
      - "tính phí"
    categories:
      - plans
      - pricing
      - billing
      - promotion
    max_chunk_size: 300
    overlap_size: 30
    
  product_b_features:
    prefix: "PB_FEAT"
    description: "Product B Features and Capabilities"
    keywords_mapping:
      - "sản phẩm b"
      - "product b"
      - "ai engine"
      - "machine learning"
      - "automation"
    categories:
      - ai
      - automation
      - analytics
      - integration
    max_chunk_size: 500
    overlap_size: 50
    
  warranty_support:
    prefix: "WARR_SUP"
    description: "Warranty and Support Information"
    keywords_mapping:
      - "bảo hành"
      - "warranty"
      - "hỗ trợ"
      - "support"
      - "chính sách"
    categories:
      - warranty
      - support
      - refund
      - policy
    max_chunk_size: 400
    overlap_size: 40
    
  contact_company:
    prefix: "CONT_INF"
    description: "Company Contact Information"
    keywords_mapping:
      - "liên hệ"
      - "contact"
      - "công ty"
      - "company"
      - "thông tin"
    categories:
      - contact
      - company
      - about
    max_chunk_size: 300
    overlap_size: 30
```

```python
# config/settings.py
import os
import yaml
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL")
    
    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    
    # FAISS
    FAISS_COLLECTIONS_PATH: str = os.getenv("FAISS_COLLECTIONS_PATH", "./faiss_collections")
    
    # Embedding
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "keepitreal/vietnamese-sbert")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", 768))
    
    # Processing
    MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", 4))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 50))
    
    # Quality Control
    MIN_WORD_COUNT: int = int(os.getenv("MIN_WORD_COUNT", 50))
    MAX_WORD_COUNT: int = int(os.getenv("MAX_WORD_COUNT", 1000))
    LANGUAGE_CONFIDENCE_THRESHOLD: float = float(os.getenv("LANGUAGE_CONFIDENCE_THRESHOLD", 0.8))
    DUPLICATE_THRESHOLD: float = float(os.getenv("DUPLICATE_THRESHOLD", 0.95))
    
    @classmethod
    def load_collections_config(cls) -> Dict[str, Any]:
        """Load collections configuration from YAML"""
        config_path = os.path.join(os.path.dirname(__file__), "collections.yaml")
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

settings = Settings()
```

## 3. Data Models

```python
# models/document.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    MARKDOWN = "markdown"
    TEXT = "text"

class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentMetadata(BaseModel):
    doc_id: str
    title: str
    collection: str
    category: str
    product: Optional[str] = None
    keywords: List[str] = Field(default_factory=list)
    auto_keywords: List[str] = Field(default_factory=list)
    source_file: str
    page_numbers: List[int] = Field(default_factory=list)
    chunk_index: int
    chunk_total: int
    word_count: int
    quality_score: float
    language: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    version: int = 1
    tags: List[str] = Field(default_factory=list)

class DocumentChunk(BaseModel):
    content: str
    metadata: DocumentMetadata
    embedding: Optional[List[float]] = None

class ParsedDocument(BaseModel):
    content: str
    title: str
    document_type: DocumentType
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunks: List[str] = Field(default_factory=list)
```

```python
# models/ingestion.py
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from .document import ProcessingStatus

class IngestionJob(BaseModel):
    job_id: str
    file_path: str
    target_collection: str
    category: Optional[str] = None
    status: ProcessingStatus = ProcessingStatus.PENDING
    total_documents: int = 0
    processed_documents: int = 0
    failed_documents: int = 0
    error_messages: List[str] = Field(default_factory=list)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    processing_time: float = 0.0
    
class IngestionResult(BaseModel):
    job_id: str
    success: bool
    documents_added: int
    documents_failed: int
    processing_time: float
    error_messages: List[str] = Field(default_factory=list)
    quality_scores: List[float] = Field(default_factory=list)
    
class BatchIngestionRequest(BaseModel):
    directory_path: str
    auto_route: bool = False
    target_collection: Optional[str] = None
    file_patterns: List[str] = Field(default=["*.pdf", "*.docx", "*.md", "*.txt"])
```

## 4. Core Processing Components

```python
# core/document_parser.py
import os
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import PyPDF2
from docx import Document as DocxDocument
import markdown
from bs4 import BeautifulSoup
from langdetect import detect, DetectorFactory

from ..models.document import ParsedDocument, DocumentType

# Đảm bảo kết quả language detection nhất quán
DetectorFactory.seed = 0

logger = logging.getLogger(__name__)

class DocumentParser:
    """Parser cho multiple file formats"""
    
    def __init__(self):
        self.supported_formats = {
            '.pdf': self._parse_pdf,
            '.docx': self._parse_docx,
            '.md': self._parse_markdown,
            '.txt': self._parse_text
        }
    
    def parse_document(self, file_path: str) -> ParsedDocument:
        """Parse document và extract content"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        extension = file_path.suffix.lower()
        if extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {extension}")
        
        logger.info(f"Parsing document: {file_path.name}")
        
        try:
            # Parse content based on file type
            parser_func = self.supported_formats[extension]
            content, metadata = parser_func(file_path)
            
            # Detect language
            try:
                language = detect(content[:1000])  # Detect từ 1000 ký tự đầu
            except:
                language = 'vi'  # Default Vietnamese
            
            # Create parsed document
            parsed_doc = ParsedDocument(
                content=content,
                title=metadata.get('title', file_path.stem),
                document_type=DocumentType(extension[1:]),
                metadata={
                    **metadata,
                    'language': language,
                    'file_size': file_path.stat().st_size,
                    'file_path': str(file_path)
                }
            )
            
            logger.info(f"Successfully parsed {file_path.name}: {len(content)} characters")
            return parsed_doc
            
        except Exception as e:
            logger.error(f"Failed to parse {file_path.name}: {e}")
            raise
    
    def _parse_pdf(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """Parse PDF file"""
        content_parts = []
        metadata = {}
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Extract metadata
                if reader.metadata:
                    metadata.update({
                        'title': reader.metadata.get('/Title', ''),
                        'author': reader.metadata.get('/Author', ''),
                        'subject': reader.metadata.get('/Subject', ''),
                        'creator': reader.metadata.get('/Creator', ''),
                    })
                
                metadata['page_count'] = len(reader.pages)
                
                # Extract text from all pages
                for page_num, page in enumerate(reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            content_parts.append(f"[Page {page_num}]\n{page_text}")
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num}: {e}")
                        continue
                
                content = '\n\n'.join(content_parts)
                
        except Exception as e:
            raise Exception(f"PDF parsing error: {e}")
        
        return content, metadata
    
    def _parse_docx(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """Parse DOCX file"""
        try:
            doc = DocxDocument(file_path)
            
            # Extract content
            content_parts = []
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    content_parts.append(paragraph.text.strip())
            
            content = '\n\n'.join(content_parts)
            
            # Extract metadata
            metadata = {}
            core_props = doc.core_properties
            if core_props:
                metadata.update({
                    'title': core_props.title or '',
                    'author': core_props.author or '',
                    'subject': core_props.subject or '',
                    'created': core_props.created,
                    'modified': core_props.modified,
                })
            
            metadata['paragraph_count'] = len(doc.paragraphs)
            
        except Exception as e:
            raise Exception(f"DOCX parsing error: {e}")
        
        return content, metadata
    
    def _parse_markdown(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """Parse Markdown file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                md_content = file.read()
            
            # Convert markdown to HTML then extract plain text
            html = markdown.markdown(md_content)
            soup = BeautifulSoup(html, 'html.parser')
            content = soup.get_text()
            
            # Basic metadata extraction
            metadata = {}
            lines = md_content.split('\n')
            
            # Try to extract title from first header
            for line in lines[:10]:
                if line.startswith('# '):
                    metadata['title'] = line[2:].strip()
                    break
            
            metadata['format'] = 'markdown'
            
        except Exception as e:
            raise Exception(f"Markdown parsing error: {e}")
        
        return content, metadata
    
    def _parse_text(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """Parse plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            metadata = {
                'format': 'text',
                'line_count': content.count('\n') + 1
            }
            
        except Exception as e:
            raise Exception(f"Text parsing error: {e}")
        
        return content, metadata
```

```python
# core/content_processor.py
import logging
import re
from typing import List, Dict, Any
from pyvi import ViTokenizer, ViPosTagger
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect

from ..models.document import ParsedDocument
from ..config.settings import settings

logger = logging.getLogger(__name__)

class ContentProcessor:
    """Xử lý và làm sạch content, chunk documents"""
    
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words=None,  # Sẽ handle Vietnamese stop words riêng
            ngram_range=(1, 2)
        )
        
        # Vietnamese stop words
        self.vi_stop_words = {
            'và', 'của', 'có', 'trong', 'là', 'với', 'được', 'các', 'để', 'cho', 
            'một', 'này', 'những', 'từ', 'theo', 'về', 'như', 'khi', 'sẽ', 'đã',
            'bởi', 'tại', 'trên', 'dưới', 'giữa', 'sau', 'trước', 'ngoài', 'trong'
        }
    
    def process_content(self, parsed_doc: ParsedDocument, 
                       max_chunk_size: int = 500, 
                       overlap_size: int = 50) -> List[str]:
        """Process và chunk document content"""
        
        logger.info(f"Processing content: {parsed_doc.title}")
        
        # Clean content
        cleaned_content = self._clean_content(parsed_doc.content)
        
        # Create chunks
        chunks = self._create_smart_chunks(
            cleaned_content, 
            max_chunk_size, 
            overlap_size
        )
        
        # Validate chunks
        valid_chunks = [chunk for chunk in chunks if self._validate_chunk(chunk)]
        
        logger.info(f"Created {len(valid_chunks)} valid chunks from {parsed_doc.title}")
        
        return valid_chunks
    
    def extract_keywords(self, content: str, top_k: int = 10) -> List[str]:
        """Extract keywords từ content using Vietnamese processing"""
        
        try:
            # Tokenize Vietnamese text
            tokenized_content = ViTokenizer.tokenize(content)
            
            # POS tagging to keep only meaningful words
            pos_tagged = ViPosTagger.postagging(tokenized_content)
            words, tags = pos_tagged
            
            # Keep only nouns, verbs, adjectives
            meaningful_tags = ['N', 'V', 'A']  # Noun, Verb, Adjective
            filtered_words = [
                word for word, tag in zip(words, tags) 
                if any(tag.startswith(mt) for mt in meaningful_tags) 
                and word.lower() not in self.vi_stop_words
                and len(word) > 2
            ]
            
            # Use TF-IDF for keyword extraction
            if len(filtered_words) > 10:  # Need minimum words for TF-IDF
                text_for_tfidf = ' '.join(filtered_words)
                try:
                    tfidf_matrix = self.tfidf_vectorizer.fit_transform([text_for_tfidf])
                    feature_names = self.tfidf_vectorizer.get_feature_names_out()
                    tfidf_scores = tfidf_matrix.toarray()[0]
                    
                    # Get top keywords
                    top_indices = tfidf_scores.argsort()[-top_k:][::-1]
                    keywords = [feature_names[i] for i in top_indices if tfidf_scores[i] > 0]
                except:
                    # Fallback to frequency-based
                    from collections import Counter
                    word_freq = Counter(filtered_words)
                    keywords = [word for word, freq in word_freq.most_common(top_k)]
            else:
                # Fallback for short content
                from collections import Counter
                word_freq = Counter(filtered_words)
                keywords = [word for word, freq in word_freq.most_common(top_k)]
            
            return keywords[:top_k]
            
        except Exception as e:
            logger.warning(f"Vietnamese keyword extraction failed, using fallback: {e}")
            return self._fallback_keyword_extraction(content, top_k)
    
    def _fallback_keyword_extraction(self, content: str, top_k: int) -> List[str]:
        """Fallback keyword extraction without Vietnamese processing"""
        words = re.findall(r'\b\w+\b', content.lower())
        words = [w for w in words if len(w) > 3 and w not in self.vi_stop_words]
        
        from collections import Counter
        word_freq = Counter(words)
        return [word for word, freq in word_freq.most_common(top_k)]
    
    def _clean_content(self, content: str) -> str:
        """Clean và normalize content"""
        
        # Remove excessive whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove special characters but keep Vietnamese
        content = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ]', ' ', content)
        
        # Remove extra spaces
        content = re.sub(r'\s+', ' ', content).strip()
        
        return content
    
    def _create_smart_chunks(self, content: str, max_size: int, overlap: int) -> List[str]:
        """Create chunks với smart splitting"""
        
        if len(content) <= max_size:
            return [content]
        
        chunks = []
        sentences = self._split_sentences(content)
        
        current_chunk = ""
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence.split())
            
            # Nếu câu quá dài, split nhỏ hơn
            if sentence_size > max_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_size = 0
                
                # Split long sentence
                sub_chunks = self._split_long_sentence(sentence, max_size)
                chunks.extend(sub_chunks)
                continue
            
            # Nếu thêm câu này vào chunk hiện tại sẽ vượt quá max_size
            if current_size + sentence_size > max_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Tạo overlap
                if overlap > 0 and chunks:
                    overlap_words = current_chunk.split()[-overlap:]
                    current_chunk = " ".join(overlap_words) + " " + sentence
                    current_size = len(overlap_words) + sentence_size
                else:
                    current_chunk = sentence
                    current_size = sentence_size
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_size += sentence_size
        
        # Add last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def _split_sentences(self, content: str) -> List[str]:
        """Split content thành sentences"""
        # Vietnamese sentence endings
        sentence_endings = r'[.!?;]\s+'
        sentences = re.split(sentence_endings, content)
        return [s.strip() for s in sentences if s.strip()]
    
    def _split_long_sentence(self, sentence: str, max_size: int) -> List[str]:
        """Split câu dài thành parts nhỏ hơn"""
        words = sentence.split()
        chunks = []
        
        for i in range(0, len(words), max_size):
            chunk = " ".join(words[i:i + max_size])
            chunks.append(chunk)
        
        return chunks
    
    def _validate_chunk(self, chunk: str) -> bool:
        """Validate chunk quality"""
        word_count = len(chunk.split())
        
        # Check word count
        if word_count < settings.MIN_WORD_COUNT or word_count > settings.MAX_WORD_COUNT:
            return False
        
        # Check if chunk has meaningful content (not just punctuation/numbers)
        meaningful_chars = re.sub(r'[\s\d\.,\!\?\;\:\-\(\)]', '', chunk)
        if len(meaningful_chars) < 10:
            return False
        
        return True
```

## 5. CLI Interface

```python
# main.py
#!/usr/bin/env python3
"""
Data Ingestion Tool CLI
Tool quản lý database cho Enterprise Chatbot System
"""

import asyncio
import click
import os
import logging
from pathlib import Path
from typing import Optional, List
import json

from core.document_parser import DocumentParser
from core.content_processor import ContentProcessor
from core.metadata_engine import MetadataEngine
from core.embedding_generator import EmbeddingGenerator
from core.collection_manager import CollectionManager
from utils.validation import DataValidator
from utils.progress_tracker import ProgressTracker
from utils.file_handler import FileHandler
from config.settings import settings

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ingestion.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@click.group()
def cli():
    """Data Ingestion Tool cho Enterprise Chatbot"""
    pass

@cli.command()
@click.option('--file', '-f', type=click.Path(exists=True), required=True, help='File path to ingest')
@click.option('--collection', '-c', required=True, help='Target collection name')
@click.option('--category', help='Document category')
@click.option('--dry-run', is_flag=True, help='Validation mode without actual ingestion')
def ingest(file: str, collection: str, category: Optional[str], dry_run: bool):
    """Ingest single document vào collection"""
    asyncio.run(_ingest_single_document(file, collection, category, dry_run))

@cli.command()
@click.option('--dir', '-d', type=click.Path(exists=True, file_okay=False), required=True, help='Directory path')
@click.option('--auto-route', is_flag=True, help='Auto-route documents to collections')
@click.option('--collection', '-c', help='Target collection (if not auto-routing)')
@click.option('--pattern', '-p', multiple=True, default=['*.pdf', '*.docx', '*.md', '*.txt'], help='File patterns')
def batch_ingest(dir: str, auto_route: bool, collection: Optional[str], pattern: List[str]):
    """Batch ingest documents từ directory"""
    asyncio.run(_batch_ingest_documents(dir, auto_route, collection, list(pattern)))

@cli.command()
@click.option('--file', '-f', type=click.Path(exists=True), required=True, help='File to validate')
@click.option('--dry-run', is_flag=True, default=True, help='Dry run mode')
def validate(file: str, dry_run: bool):
    """Validate document quality trước khi ingest"""
    asyncio.run(_validate_document(file, dry_run))

@cli.command()
def list_collections():
    """List tất cả collections available"""
    asyncio.run(_list_collections())

@cli.command()
@click.option('--name', '-n', required=True, help='Collection name')
def collection_stats(name: str):
    """Show statistics for collection"""
    asyncio.run(_show_collection_stats(name))

@cli.command()
@click.option('--collection', '-c', help='Collection name (or "all" for all collections)')
@click.option('--min-score', type=float, default=0.7, help='Minimum quality score')
def quality_check(collection: str, min_score: float):
    """Check quality scores of documents"""
    asyncio.run(_quality_check(collection, min_score))

@cli.command()
@click.option('--tail', '-t', type=int, default=50, help='Number of log lines to show')
def logs(tail: int):
    """Show recent logs"""
    _show_logs(tail)

async def _ingest_single_document(file_path: str, collection: str, category: Optional[str], dry_run: bool):
    """Ingest single document"""
    try:
        click.echo(f"🚀 Ingesting document: {file_path}")
        
        # Initialize components
        parser = DocumentParser()
        processor = ContentProcessor()
        metadata_engine = MetadataEngine()
        embedding_generator = EmbeddingGenerator()
        collection_manager = CollectionManager()
        validator = DataValidator()
        
        # Parse document
        click.echo("📄 Parsing document...")
        parsed_doc = parser.parse_document(file_path)
        
        # Process content
        click.echo("⚙️ Processing content...")
        chunks = processor.process_content(parsed_doc)
        
        if not chunks:
            click.echo("❌ No valid chunks created from document")
            return
        
        click.echo(f"📝 Created {len(chunks)} chunks")
        
        # Generate metadata
        click.echo("🏷️ Generating metadata...")
        document_chunks = []
        
        for i, chunk_content in enumerate(chunks):
            metadata = await metadata_engine.generate_metadata(
                content=chunk_content,
                title=parsed_doc.title,
                collection=collection,
                category=category,
                chunk_index=i,
                chunk_total=len(chunks),
                source_file=os.path.basename(file_path)
            )
            
            # Validate chunk
            if not validator.validate_chunk_quality(chunk_content, metadata):
                click.echo(f"⚠️ Chunk {i+1} failed quality validation, skipping...")
                continue
            
            # Generate embeddings
            if not dry_run:
                click.echo(f"🧮 Generating embeddings for chunk {i+1}...")
                embedding = await embedding_generator.generate_embedding(chunk_content)
                metadata.embedding = embedding
            
            from models.document import DocumentChunk
            doc_chunk = DocumentChunk(
                content=chunk_content,
                metadata=metadata
            )
            document_chunks.append(doc_chunk)
        
        if not document_chunks:
            click.echo("❌ No valid document chunks after processing")
            return
        
        click.echo(f"✅ Processed {len(document_chunks)} valid chunks")
        
        if dry_run:
            click.echo("🔍 DRY RUN MODE - No data will be saved")
            click.echo(f"Would add {len(document_chunks)} chunks to collection '{collection}'")
            return
        
        # Save to collection
        click.echo(f"💾 Saving to collection '{collection}'...")
        added_count = await collection_manager.add_documents_to_collection(
            collection, document_chunks
        )
        
        click.echo(f"🎉 Successfully added {added_count} documents to collection '{collection}'")
        
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        click.echo(
Tiếp tục code từ CLI interface:

```python
        click.echo(f"❌ Ingestion failed: {e}")
        
async def _batch_ingest_documents(dir_path: str, auto_route: bool, collection: Optional[str], patterns: List[str]):
    """Batch ingest documents from directory"""
    try:
        click.echo(f"📁 Scanning directory: {dir_path}")
        
        file_handler = FileHandler()
        files = file_handler.find_documents(dir_path, patterns)
        
        if not files:
            click.echo("No matching files found")
            return
        
        click.echo(f"Found {len(files)} documents to process")
        
        # Initialize components
        parser = DocumentParser()
        processor = ContentProcessor()
        metadata_engine = MetadataEngine()
        embedding_generator = EmbeddingGenerator()
        collection_manager = CollectionManager()
        
        progress = ProgressTracker(total=len(files))
        
        successful = 0
        failed = 0
        
        for file_path in files:
            try:
                click.echo(f"Processing: {os.path.basename(file_path)}")
                
                # Auto-route collection if enabled
                target_collection = collection
                if auto_route:
                    target_collection = await _auto_route_collection(file_path)
                
                if not target_collection:
                    click.echo(f"❌ Could not determine collection for {file_path}")
                    failed += 1
                    continue
                
                # Process single file (reuse logic from _ingest_single_document)
                result = await _process_single_file(
                    file_path, target_collection, parser, processor, 
                    metadata_engine, embedding_generator, collection_manager
                )
                
                if result:
                    successful += 1
                    click.echo(f"✅ {os.path.basename(file_path)} -> {target_collection}")
                else:
                    failed += 1
                    click.echo(f"❌ Failed: {os.path.basename(file_path)}")
                
                progress.update(1)
                
            except Exception as e:
                failed += 1
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        click.echo(f"\n📊 Batch ingestion completed:")
        click.echo(f"✅ Successful: {successful}")
        click.echo(f"❌ Failed: {failed}")
        
    except Exception as e:
        click.echo(f"❌ Batch ingestion failed: {e}")

async def _process_single_file(file_path: str, collection: str, parser, processor, 
                              metadata_engine, embedding_generator, collection_manager):
    """Process single file (helper function)"""
    try:
        # Parse
        parsed_doc = parser.parse_document(file_path)
        
        # Process content
        chunks = processor.process_content(parsed_doc)
        if not chunks:
            return False
        
        # Create document chunks
        document_chunks = []
        for i, chunk_content in enumerate(chunks):
            metadata = await metadata_engine.generate_metadata(
                content=chunk_content,
                title=parsed_doc.title,
                collection=collection,
                chunk_index=i,
                chunk_total=len(chunks),
                source_file=os.path.basename(file_path)
            )
            
            embedding = await embedding_generator.generate_embedding(chunk_content)
            
            from models.document import DocumentChunk
            doc_chunk = DocumentChunk(
                content=chunk_content,
                metadata=metadata,
                embedding=embedding
            )
            document_chunks.append(doc_chunk)
        
        # Save to collection
        added_count = await collection_manager.add_documents_to_collection(
            collection, document_chunks
        )
        
        return added_count > 0
        
    except Exception as e:
        logger.error(f"Process single file error: {e}")
        return False

async def _auto_route_collection(file_path: str) -> Optional[str]:
    """Auto-determine target collection based on file content"""
    try:
        parser = DocumentParser()
        parsed_doc = parser.parse_document(file_path)
        
        # Simple keyword-based routing
        content_lower = parsed_doc.content.lower()
        
        collections_config = settings.load_collections_config()['collections']
        
        max_score = 0
        best_collection = None
        
        for collection_name, config in collections_config.items():
            score = 0
            for keyword in config.get('keywords_mapping', []):
                if keyword.lower() in content_lower:
                    score += content_lower.count(keyword.lower())
            
            if score > max_score:
                max_score = score
                best_collection = collection_name
        
        return best_collection if max_score > 0 else None
        
    except Exception as e:
        logger.error(f"Auto-route error: {e}")
        return None

async def _validate_document(file_path: str, dry_run: bool):
    """Validate document before ingestion"""
    try:
        click.echo(f"🔍 Validating document: {file_path}")
        
        validator = DataValidator()
        parser = DocumentParser()
        processor = ContentProcessor()
        
        # Parse document
        parsed_doc = parser.parse_document(file_path)
        
        # Basic validation
        file_valid = validator.validate_file(file_path)
        if not file_valid:
            click.echo("❌ File validation failed")
            return
        
        # Content validation
        content_valid = validator.validate_content(parsed_doc.content)
        if not content_valid:
            click.echo("❌ Content validation failed")
            return
        
        # Process chunks
        chunks = processor.process_content(parsed_doc)
        
        valid_chunks = 0
        total_chunks = len(chunks)
        
        for chunk in chunks:
            if validator.validate_chunk_quality(chunk, None):
                valid_chunks += 1
        
        click.echo(f"📊 Validation Results:")
        click.echo(f"  Total chunks: {total_chunks}")
        click.echo(f"  Valid chunks: {valid_chunks}")
        click.echo(f"  Success rate: {(valid_chunks/total_chunks)*100:.1f}%")
        
        if valid_chunks == 0:
            click.echo("❌ No valid chunks found")
        elif valid_chunks < total_chunks * 0.8:
            click.echo("⚠️  Low quality document - many chunks failed validation")
        else:
            click.echo("✅ Document validation passed")
        
    except Exception as e:
        click.echo(f"❌ Validation failed: {e}")

async def _list_collections():
    """List all available collections"""
    try:
        collections_config = settings.load_collections_config()['collections']
        
        click.echo("📚 Available Collections:")
        click.echo("-" * 50)
        
        for name, config in collections_config.items():
            click.echo(f"  {name}")
            click.echo(f"    Description: {config.get('description', 'N/A')}")
            click.echo(f"    Prefix: {config.get('prefix', 'N/A')}")
            click.echo(f"    Categories: {', '.join(config.get('categories', []))}")
            click.echo()
        
    except Exception as e:
        click.echo(f"❌ Failed to list collections: {e}")

async def _show_collection_stats(collection_name: str):
    """Show statistics for specific collection"""
    try:
        collection_manager = CollectionManager()
        stats = await collection_manager.get_collection_stats(collection_name)
        
        click.echo(f"📊 Statistics for collection: {collection_name}")
        click.echo("-" * 50)
        click.echo(f"Total documents: {stats.get('document_count', 0)}")
        click.echo(f"Total chunks: {stats.get('chunk_count', 0)}")
        click.echo(f"Index size: {stats.get('index_size', 0)}")
        click.echo(f"Average quality score: {stats.get('avg_quality_score', 0):.2f}")
        click.echo(f"Last updated: {stats.get('last_updated', 'N/A')}")
        
    except Exception as e:
        click.echo(f"❌ Failed to get collection stats: {e}")

async def _quality_check(collection: str, min_score: float):
    """Check document quality scores"""
    try:
        collection_manager = CollectionManager()
        
        if collection == "all":
            collections_config = settings.load_collections_config()['collections']
            collections_to_check = list(collections_config.keys())
        else:
            collections_to_check = [collection]
        
        for coll_name in collections_to_check:
            click.echo(f"🔍 Quality check for: {coll_name}")
            
            quality_results = await collection_manager.check_quality_scores(
                coll_name, min_score
            )
            
            total = quality_results.get('total_documents', 0)
            below_threshold = quality_results.get('below_threshold', 0)
            
            if total > 0:
                percentage = (below_threshold / total) * 100
                click.echo(f"  Documents below {min_score}: {below_threshold}/{total} ({percentage:.1f}%)")
            else:
                click.echo(f"  No documents found in collection")
            
            click.echo()
        
    except Exception as e:
        click.echo(f"❌ Quality check failed: {e}")

def _show_logs(tail: int):
    """Show recent log entries"""
    try:
        log_file = "ingestion.log"
        if not os.path.exists(log_file):
            click.echo("No log file found")
            return
        
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        recent_lines = lines[-tail:] if len(lines) > tail else lines
        
        click.echo(f"📋 Last {len(recent_lines)} log entries:")
        click.echo("-" * 50)
        
        for line in recent_lines:
            click.echo(line.rstrip())
        
    except Exception as e:
        click.echo(f"❌ Failed to show logs: {e}")

if __name__ == "__main__":
    cli()
```

## 6. Metadata Engine

```python
# core/metadata_engine.py
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional
from pyvi import ViTokenizer

from ..models.document import DocumentMetadata
from ..core.content_processor import ContentProcessor
from ..config.settings import settings

logger = logging.getLogger(__name__)

class MetadataEngine:
    """Engine để generate metadata cho documents"""
    
    def __init__(self):
        self.content_processor = ContentProcessor()
        self.collections_config = settings.load_collections_config()['collections']
    
    async def generate_metadata(self, 
                              content: str,
                              title: str,
                              collection: str,
                              category: Optional[str] = None,
                              chunk_index: int = 0,
                              chunk_total: int = 1,
                              source_file: str = "",
                              product: Optional[str] = None) -> DocumentMetadata:
        """Generate complete metadata cho document chunk"""
        
        try:
            # Get collection config
            collection_config = self.collections_config.get(collection, {})
            
            # Generate document ID
            doc_id = self._generate_doc_id(collection, chunk_index)
            
            # Auto-detect category nếu chưa có
            if not category:
                category = await self._detect_category(content, collection)
            
            # Auto-detect product nếu chưa có
            if not product:
                product = await self._detect_product(content, collection)
            
            # Extract keywords
            manual_keywords = self._extract_manual_keywords(content, collection)
            auto_keywords = self.content_processor.extract_keywords(content, top_k=8)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(content, auto_keywords)
            
            # Word count
            word_count = len(content.split())
            
            # Language detection
            language = self._detect_language(content)
            
            metadata = DocumentMetadata(
                doc_id=doc_id,
                title=title,
                collection=collection,
                category=category or "general",
                product=product,
                keywords=manual_keywords,
                auto_keywords=auto_keywords,
                source_file=source_file,
                chunk_index=chunk_index,
                chunk_total=chunk_total,
                word_count=word_count,
                quality_score=quality_score,
                language=language,
                created_at=datetime.now(),
                updated_at=datetime.now(),
                version=1,
                tags=self._generate_tags(content, collection, category)
            )
            
            logger.info(f"Generated metadata for {doc_id}: quality={quality_score:.2f}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to generate metadata: {e}")
            raise
    
    def _generate_doc_id(self, collection: str, chunk_index: int) -> str:
        """Generate unique document ID"""
        collection_config = self.collections_config.get(collection, {})
        prefix = collection_config.get('prefix', 'DOC')
        
        # Get sequential number (trong production sẽ query database)
        sequential_id = self._get_next_sequential_id(collection)
        
        doc_id = f"{prefix}_{sequential_id:03d}_v1"
        if chunk_index > 0:
            doc_id = f"{prefix}_{sequential_id:03d}_{chunk_index}_v1"
        
        return doc_id
    
    def _get_next_sequential_id(self, collection: str) -> int:
        """Get next sequential ID cho collection"""
        # Simplified - trong production sẽ query database để get max ID
        return len(str(uuid.uuid4())) % 1000 + 1
    
    async def _detect_category(self, content: str, collection: str) -> str:
        """Auto-detect category dựa trên content"""
        collection_config = self.collections_config.get(collection, {})
        categories = collection_config.get('categories', [])
        
        if not categories:
            return "general"
        
        content_lower = content.lower()
        
        # Category keyword mapping
        category_keywords = {
            'security': ['bảo mật', 'security', 'authentication', 'mã hóa', 'ssl', 'tls'],
            'pricing': ['giá', 'price', 'chi phí', 'cost', 'gói', 'plan'],
            'integration': ['tích hợp', 'integration', 'api', 'connect', 'kết nối'],
            'support': ['hỗ trợ', 'support', 'help', 'giúp đỡ'],
            'warranty': ['bảo hành', 'warranty', 'chính sách', 'policy'],
            'contact': ['liên hệ', 'contact', 'thông tin', 'info'],
            'analytics': ['phân tích', 'analytics', 'báo cáo', 'report'],
            'performance': ['hiệu suất', 'performance', 'tốc độ', 'speed'],
            'ai': ['ai', 'artificial intelligence', 'machine learning', 'trí tuệ nhân tạo'],
            'automation': ['tự động', 'automation', 'workflow']
        }
        
        category_scores = {}
        
        for category in categories:
            score = 0
            keywords = category_keywords.get(category, [])
            
            for keyword in keywords:
                score += content_lower.count(keyword)
            
            category_scores[category] = score
        
        # Return category với score cao nhất
        if category_scores:
            best_category = max(category_scores.items(), key=lambda x: x[1])[0]
            if category_scores[best_category] > 0:
                return best_category
        
        return "general"
    
    async def _detect_product(self, content: str, collection: str) -> Optional[str]:
        """Auto-detect product từ content"""
        content_lower = content.lower()
        
        product_keywords = {
            'product_a': ['sản phẩm a', 'product a'],
            'product_b': ['sản phẩm b', 'product b']
        }
        
        for product, keywords in product_keywords.items():
            for keyword in keywords:
                if keyword in content_lower:
                    return product
        
        # Detect từ collection name
        if 'product_a' in collection:
            return 'product_a'
        elif 'product_b' in collection:
            return 'product_b'
        
        return None
    
    def _extract_manual_keywords(self, content: str, collection: str) -> List[str]:
        """Extract manual keywords dựa trên collection config"""
        collection_config = self.collections_config.get(collection, {})
        keyword_mapping = collection_config.get('keywords_mapping', [])
        
        content_lower = content.lower()
        found_keywords = []
        
        for keyword in keyword_mapping:
            if keyword.lower() in content_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _calculate_quality_score(self, content: str, keywords: List[str]) -> float:
        """Calculate quality score cho content"""
        score = 0.5  # Base score
        
        # Word count factor
        word_count = len(content.split())
        if settings.MIN_WORD_COUNT <= word_count <= settings.MAX_WORD_COUNT:
            score += 0.2
        elif word_count < settings.MIN_WORD_COUNT:
            score -= 0.2
        
        # Keywords factor
        if len(keywords) > 0:
            score += min(len(keywords) * 0.05, 0.2)
        
        # Content coherence (simple check)
        sentences = content.split('.')
        if len(sentences) > 1:
            score += 0.1
        
        # Vietnamese text check
        vietnamese_chars = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
        has_vietnamese = any(char in content.lower() for char in vietnamese_chars)
        if has_vietnamese:
            score += 0.1
        
        return min(max(score, 0.0), 1.0)  # Clamp giữa 0 và 1
    
    def _detect_language(self, content: str) -> str:
        """Detect language của content"""
        try:
            from langdetect import detect
            detected = detect(content)
            return detected if detected in ['vi', 'en'] else 'vi'
        except:
            # Fallback detection
            vietnamese_chars = 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ'
            has_vietnamese = any(char in content.lower() for char in vietnamese_chars)
            return 'vi' if has_vietnamese else 'en'
    
    def _generate_tags(self, content: str, collection: str, category: str) -> List[str]:
        """Generate tags cho document"""
        tags = []
        
        # Add category tag
        if category:
            tags.append(category)
        
        # Add collection-based tags
        if 'features' in collection:
            tags.append('feature')
        if 'pricing' in collection:
            tags.append('pricing')
        if 'support' in collection or 'warranty' in collection:
            tags.append('support')
        
        # Content-based tags
        content_lower = content.lower()
        if any(word in content_lower for word in ['technical', 'api', 'integration']):
            tags.append('technical')
        if any(word in content_lower for word in ['customer', 'user', 'client']):
            tags.append('customer-facing')
        
        return list(set(tags))  # Remove duplicates
```

## 7. Embedding Generator

```python
# core/embedding_generator.py
import logging
import asyncio
from typing import List, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from ..config.settings import settings

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings cho Vietnamese text"""
    
    def __init__(self):
        self.model = None
        self.model_name = settings.EMBEDDING_MODEL
        self.dimension = settings.EMBEDDING_DIMENSION
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize embedding model"""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            
            # Sử dụng model tối ưu cho tiếng Việt
            self.model = SentenceTransformer(self.model_name)
            
            # Set device
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = self.model.to(device)
            
            logger.info(f"Embedding model loaded on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            # Fallback to basic model
            try:
                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                logger.info("Using fallback embedding model")
            except Exception as fallback_e:
                logger.error(f"Fallback model also failed: {fallback_e}")
                raise
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding cho single text"""
        try:
            if not self.model:
                raise Exception("Embedding model not initialized")
            
            # Preprocess text
            text = self._preprocess_text(text)
            
            # Generate embedding
            embedding = self.model.encode(text, convert_to_tensor=False)
            
            # Convert to list nếu cần
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    async def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings cho batch texts"""
        try:
            if not self.model:
                raise Exception("Embedding model not initialized")
            
            # Preprocess texts
            processed_texts = [self._preprocess_text(text) for text in texts]
            
            # Generate embeddings in batch
            embeddings = self.model.encode(processed_texts, convert_to_tensor=False)
            
            # Convert to list format
            if isinstance(embeddings, np.ndarray):
                embeddings = embeddings.tolist()
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            raise
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text trước khi generate embedding"""
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        
        # Truncate nếu quá dài (model limits)
        max_length = 512  # Typical transformer limit
        words = text.split()
        if len(words) > max_length:
            text = ' '.join(words[:max_length])
        
        return text
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension"""
        return self.dimension
    
    async def similarity_search(self, query_embedding: List[float], 
                               document_embeddings: List[List[float]], 
                               top_k: int = 5) -> List[tuple]:
        """Search similar documents dựa trên embeddings"""
        try:
            query_vector = np.array(query_embedding)
            doc_vectors = np.array(document_embeddings)
            
            # Cosine similarity
            similarities = np.dot(doc_vectors, query_vector) / (
                np.linalg.norm(doc_vectors, axis=1) * np.linalg.norm(query_vector)
            )
            
            # Get top k indices và scores
            top_indices = np.argsort(similarities)[::-1][:top_k]
            top_scores = similarities[top_indices]
            
            results = [(int(idx), float(score)) for idx, score in zip(top_indices, top_scores)]
            return results
            
        except Exception as e:
            logger.error(f"Similarity search failed: {e}")
            return []
    
    async def health_check(self) -> dict:
        """Health check cho embedding service"""
        try:
            if not self.model:
                return {"status": "unhealthy", "error": "Model not loaded"}
            
            # Test embedding generation
            test_text = "Test embedding generation"
            embedding = await self.generate_embedding(test_text)
            
            return {
                "status": "healthy",
                "model": self.model_name,
                "dimension": len(embedding),
                "device": str(self.model.device) if hasattr(self.model, 'device') else "unknown"
            }
            
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
```

## 8. Collection Manager

```python
# core/collection_manager.py
import logging
import os
import pickle
import asyncio
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from datetime import datetime

from ..models.document import DocumentChunk
from ..config.settings import settings

logger = logging.getLogger(__name__)

class CollectionManager:
    """Quản lý FAISS collections và document storage"""
    
    def __init__(self):
        self.collections_path = settings.FAISS_COLLECTIONS_PATH
        self.collections: Dict[str, Any] = {}
        self.indexes: Dict[str, faiss.IndexFlatIP] = {}  # Inner product for cosine similarity
        self.document_metadata: Dict[str, List[Dict]] = {}
        
        # Ensure collections directory exists
        os.makedirs(self.collections_path, exist_ok=True)
    
    async def initialize_collections(self):
        """Initialize tất cả collections"""
        try:
            collections_config = settings.load_collections_config()['collections']
            
            for collection_name in collections_config.keys():
                await self._initialize_collection(collection_name)
            
            logger.info(f"Initialized {len(self.collections)} collections")
            
        except Exception as e:
            logger.error(f"Failed to initialize collections: {e}")
            raise
    
    async def _initialize_collection(self, collection_name: str):
        """Initialize single collection"""
        try:
            collection_dir = os.path.join(self.collections_path, collection_name)
            os.makedirs(collection_dir, exist_ok=True)
            
            index_file = os.path.join(collection_dir, "index.faiss")
            metadata_file = os.path.join(collection_dir, "metadata.pkl")
            
            # Load existing index nếu có
            if os.path.exists(index_file):
                index = faiss.read_index(index_file)
                logger.info(f"Loaded existing index for {collection_name}: {index.ntotal} documents")
            else:
                # Create new index
                dimension = settings.EMBEDDING_DIMENSION
                index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
                logger.info(f"Created new index for {collection_name}")
            
            self.indexes[collection_name] = index
            
            # Load metadata
            if os.path.exists(metadata_file):
                with open(metadata_file, 'rb') as f:
                    metadata = pickle.load(f)
                self.document_metadata[collection_name] = metadata
                logger.info(f"Loaded metadata for {collection_name}: {len(metadata)} documents")
            else:
                self.document_metadata[collection_name] = []
            
            self.collections[collection_name] = {
                'index': index,
                'metadata': self.document_metadata[collection_name],
                'last_updated': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize collection {collection_name}: {e}")
            raise
    
    async def add_documents_to_collection(self, collection_name: str, 
                                        document_chunks: List[DocumentChunk]) -> int:
        """Add documents to collection"""
        try:
            if collection_name not in self.collections:
                await self._initialize_collection(collection_name)
            
            index = self.indexes[collection_name]
            metadata_list = self.document_metadata[collection_name]
            
            # Prepare embeddings và metadata
            embeddings = []
            metadata_batch = []
            
            for chunk in document_chunks:
                if not chunk.embedding:
                    logger.warning(f"Document chunk missing embedding: {chunk.metadata.doc_id}")
                    continue
                
                embeddings.append(chunk.embedding)
                
                # Convert metadata to dict for storage
                metadata_dict = {
                    'doc_id': chunk.metadata.doc_id,
                    'title': chunk.metadata.title,
                    'content': chunk.content,
                    'category': chunk.metadata.category,
                    'product': chunk.metadata.product,
                    'keywords': chunk.metadata.keywords,
                    'auto_keywords': chunk.metadata.auto_keywords,
                    'quality_score': chunk.metadata.quality_score,
                    'word_count': chunk.metadata.word_count,
                    'chunk_index': chunk.metadata.chunk_index,
                    'source_file': chunk.metadata.source_file,
                    'created_at': chunk.metadata.created_at.isoformat(),
                    'tags': chunk.metadata.tags
                }
                metadata_batch.append(metadata_dict)
            
            if not embeddings:
                logger.warning("No valid embeddings to add")
                return 0
            
            # Normalize embeddings cho cosine similarity
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            
            # Add to FAISS index
            index.add(embeddings_array)
            
            # Add metadata
            metadata_list.extend(metadata_batch)
            
            # Save to disk
            await self._save_collection(collection_name)
            
            # Update collection info
            self.collections[collection_name]['last_updated'] = datetime.now()
            
            logger.info(f"Added {len(embeddings)} documents to collection {collection_name}")
            return len(embeddings)
            
        except Exception as e:
            logger.error(f"Failed to add documents
Tiếp tục code từ Collection Manager:

```python
            logger.error(f"Failed to add documents to collection {collection_name}: {e}")
            raise
    
    async def search_collection(self, collection_name: str, query_embedding: List[float], 
                              top_k: int = 5, filters: Dict[str, Any] = None) -> List[Dict]:
        """Search documents trong collection"""
        try:
            if collection_name not in self.collections:
                logger.warning(f"Collection {collection_name} not found")
                return []
            
            index = self.indexes[collection_name]
            metadata_list = self.document_metadata[collection_name]
            
            if index.ntotal == 0:
                return []
            
            # Normalize query embedding
            query_vector = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_vector)
            
            # Search
            scores, indices = index.search(query_vector, min(top_k, index.ntotal))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(metadata_list):
                    doc_metadata = metadata_list[idx].copy()
                    doc_metadata['similarity_score'] = float(score)
                    
                    # Apply filters nếu có
                    if filters and not self._apply_filters(doc_metadata, filters):
                        continue
                    
                    results.append(doc_metadata)
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed for collection {collection_name}: {e}")
            return []
    
    def _apply_filters(self, metadata: Dict, filters: Dict[str, Any]) -> bool:
        """Apply filters lên document metadata"""
        for filter_key, filter_value in filters.items():
            if filter_key not in metadata:
                continue
            
            doc_value = metadata[filter_key]
            
            if isinstance(filter_value, list):
                if doc_value not in filter_value:
                    return False
            elif isinstance(filter_value, str):
                if filter_value.lower() not in str(doc_value).lower():
                    return False
            else:
                if doc_value != filter_value:
                    return False
        
        return True
    
    async def _save_collection(self, collection_name: str):
        """Save collection to disk"""
        try:
            collection_dir = os.path.join(self.collections_path, collection_name)
            
            index_file = os.path.join(collection_dir, "index.faiss")
            metadata_file = os.path.join(collection_dir, "metadata.pkl")
            
            # Save FAISS index
            index = self.indexes[collection_name]
            faiss.write_index(index, index_file)
            
            # Save metadata
            metadata = self.document_metadata[collection_name]
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
        except Exception as e:
            logger.error(f"Failed to save collection {collection_name}: {e}")
            raise
    
    async def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """Get statistics cho collection"""
        try:
            if collection_name not in self.collections:
                return {"error": "Collection not found"}
            
            index = self.indexes[collection_name]
            metadata_list = self.document_metadata[collection_name]
            
            # Calculate statistics
            total_documents = len(metadata_list)
            quality_scores = [doc.get('quality_score', 0) for doc in metadata_list]
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            # Index size estimation
            index_size = index.ntotal * settings.EMBEDDING_DIMENSION * 4  # 4 bytes per float
            
            return {
                'document_count': total_documents,
                'chunk_count': index.ntotal,
                'index_size': index_size,
                'avg_quality_score': avg_quality,
                'last_updated': self.collections[collection_name]['last_updated'].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats for {collection_name}: {e}")
            return {"error": str(e)}
    
    async def check_quality_scores(self, collection_name: str, min_score: float) -> Dict[str, int]:
        """Check quality scores của documents"""
        try:
            if collection_name not in self.collections:
                return {"error": "Collection not found"}
            
            metadata_list = self.document_metadata[collection_name]
            
            total_documents = len(metadata_list)
            below_threshold = 0
            
            for doc in metadata_list:
                quality_score = doc.get('quality_score', 0)
                if quality_score < min_score:
                    below_threshold += 1
            
            return {
                'total_documents': total_documents,
                'below_threshold': below_threshold,
                'percentage': (below_threshold / total_documents * 100) if total_documents > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Quality check failed for {collection_name}: {e}")
            return {"error": str(e)}
    
    async def remove_document(self, collection_name: str, doc_id: str) -> bool:
        """Remove document từ collection"""
        try:
            if collection_name not in self.collections:
                return False
            
            metadata_list = self.document_metadata[collection_name]
            
            # Find document index
            doc_index = None
            for i, doc in enumerate(metadata_list):
                if doc.get('doc_id') == doc_id:
                    doc_index = i
                    break
            
            if doc_index is None:
                logger.warning(f"Document {doc_id} not found in {collection_name}")
                return False
            
            # Remove từ metadata list
            metadata_list.pop(doc_index)
            
            # FAISS không support remove individual vectors efficiently
            # Cần rebuild index
            await self._rebuild_collection_index(collection_name)
            
            logger.info(f"Removed document {doc_id} from {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove document {doc_id}: {e}")
            return False
    
    async def _rebuild_collection_index(self, collection_name: str):
        """Rebuild FAISS index cho collection"""
        try:
            metadata_list = self.document_metadata[collection_name]
            
            # Create new index
            dimension = settings.EMBEDDING_DIMENSION
            new_index = faiss.IndexFlatIP(dimension)
            
            # Re-add all embeddings
            if metadata_list:
                # Note: Trong implementation thật, cần store embeddings riêng
                # Đây là simplified version
                logger.warning(f"Index rebuild cho {collection_name} - embeddings cần được re-generated")
            
            self.indexes[collection_name] = new_index
            await self._save_collection(collection_name)
            
        except Exception as e:
            logger.error(f"Failed to rebuild index for {collection_name}: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check cho collections"""
        try:
            collections_status = {}
            
            for name, collection in self.collections.items():
                index = self.indexes[name]
                metadata_count = len(self.document_metadata[name])
                
                collections_status[name] = {
                    'documents': metadata_count,
                    'index_size': index.ntotal,
                    'last_updated': collection['last_updated'].isoformat()
                }
            
            return {
                'status': 'healthy',
                'collections': collections_status,
                'total_collections': len(self.collections)
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e)
            }
```

## 9. Validation & Utils

```python
# utils/validation.py
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import magic  # python-magic for file type detection

from ..config.settings import settings
from ..models.document import DocumentMetadata

logger = logging.getLogger(__name__)

class DataValidator:
    """Data validation cho ingestion process"""
    
    def __init__(self):
        self.supported_extensions = ['.pdf', '.docx', '.md', '.txt']
        self.max_file_size = 50 * 1024 * 1024  # 50MB
    
    def validate_file(self, file_path: str) -> bool:
        """Validate file trước khi processing"""
        try:
            file_path = Path(file_path)
            
            # Check file exists
            if not file_path.exists():
                logger.error(f"File does not exist: {file_path}")
                return False
            
            # Check file extension
            if file_path.suffix.lower() not in self.supported_extensions:
                logger.error(f"Unsupported file type: {file_path.suffix}")
                return False
            
            # Check file size
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                logger.error(f"File too large: {file_size} bytes")
                return False
            
            if file_size == 0:
                logger.error("File is empty")
                return False
            
            # Check file type (MIME)
            try:
                mime_type = magic.from_file(str(file_path), mime=True)
                expected_types = {
                    '.pdf': 'application/pdf',
                    '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    '.md': 'text/plain',
                    '.txt': 'text/plain'
                }
                
                extension = file_path.suffix.lower()
                if extension in expected_types:
                    if not (mime_type == expected_types[extension] or 
                           (extension in ['.md', '.txt'] and mime_type.startswith('text/'))):
                        logger.warning(f"MIME type mismatch: expected {expected_types[extension]}, got {mime_type}")
                        # Warning only, not blocking
            
            except ImportError:
                logger.warning("python-magic not available, skipping MIME type check")
            except Exception as e:
                logger.warning(f"MIME type check failed: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"File validation error: {e}")
            return False
    
    def validate_content(self, content: str) -> bool:
        """Validate document content"""
        try:
            # Check empty content
            if not content or not content.strip():
                logger.error("Content is empty")
                return False
            
            # Check minimum content length
            if len(content.strip()) < 10:
                logger.error("Content too short")
                return False
            
            # Check for reasonable text content
            # Tính percentage của printable characters
            printable_chars = sum(1 for char in content if char.isprintable() or char.isspace())
            printable_ratio = printable_chars / len(content) if content else 0
            
            if printable_ratio < 0.8:
                logger.error(f"Content has too many non-printable characters: {printable_ratio:.2f}")
                return False
            
            # Check language detection có success không
            try:
                from langdetect import detect
                detected_lang = detect(content[:1000])  # Check first 1000 chars
                if detected_lang not in ['vi', 'en', 'zh', 'ja', 'ko']:  # Common languages
                    logger.warning(f"Detected language: {detected_lang}")
            except:
                logger.warning("Language detection failed")
            
            return True
            
        except Exception as e:
            logger.error(f"Content validation error: {e}")
            return False
    
    def validate_chunk_quality(self, chunk_content: str, metadata: Optional[DocumentMetadata]) -> bool:
        """Validate individual chunk quality"""
        try:
            # Word count validation
            words = chunk_content.split()
            word_count = len(words)
            
            if word_count < settings.MIN_WORD_COUNT:
                logger.debug(f"Chunk too short: {word_count} words")
                return False
            
            if word_count > settings.MAX_WORD_COUNT:
                logger.debug(f"Chunk too long: {word_count} words")
                return False
            
            # Content coherence check
            sentences = [s.strip() for s in chunk_content.split('.') if s.strip()]
            if len(sentences) < 2:
                logger.debug("Chunk has too few sentences")
                return False
            
            # Check for meaningful content
            meaningful_chars = sum(1 for char in chunk_content 
                                 if char.isalnum() or char in 'áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ')
            
            if meaningful_chars < len(chunk_content) * 0.5:
                logger.debug("Chunk lacks meaningful content")
                return False
            
            # Quality score check nếu có metadata
            if metadata and hasattr(metadata, 'quality_score'):
                if metadata.quality_score < 0.3:
                    logger.debug(f"Chunk quality score too low: {metadata.quality_score}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Chunk validation error: {e}")
            return False
    
    def validate_metadata(self, metadata: DocumentMetadata) -> bool:
        """Validate document metadata"""
        try:
            # Required fields
            required_fields = ['doc_id', 'title', 'collection', 'category']
            for field in required_fields:
                if not getattr(metadata, field, None):
                    logger.error(f"Missing required metadata field: {field}")
                    return False
            
            # Doc ID format
            if not self._validate_doc_id_format(metadata.doc_id):
                logger.error(f"Invalid doc_id format: {metadata.doc_id}")
                return False
            
            # Quality score range
            if not (0 <= metadata.quality_score <= 1):
                logger.error(f"Invalid quality score: {metadata.quality_score}")
                return False
            
            # Word count
            if metadata.word_count <= 0:
                logger.error(f"Invalid word count: {metadata.word_count}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Metadata validation error: {e}")
            return False
    
    def _validate_doc_id_format(self, doc_id: str) -> bool:
        """Validate document ID format"""
        # Format: {PREFIX}_{NUMBER}_v{VERSION} or {PREFIX}_{NUMBER}_{CHUNK}_v{VERSION}
        import re
        
        patterns = [
            r'^[A-Z_]+_\d{3}_v\d+$',  # PA_FEAT_001_v1
            r'^[A-Z_]+_\d{3}_\d+_v\d+$'  # PA_FEAT_001_1_v1
        ]
        
        return any(re.match(pattern, doc_id) for pattern in patterns)
    
    def check_duplicates(self, content: str, existing_contents: list) -> list:
        """Check for duplicate content"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            
            if not existing_contents:
                return []
            
            # Prepare texts
            texts = [content] + existing_contents
            
            # Vectorize
            vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Calculate similarity
            similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
            
            # Find high similarity matches
            duplicates = []
            for i, similarity in enumerate(similarities):
                if similarity > settings.DUPLICATE_THRESHOLD:
                    duplicates.append({
                        'index': i,
                        'similarity': float(similarity),
                        'content_preview': existing_contents[i][:100] + "..."
                    })
            
            return duplicates
            
        except Exception as e:
            logger.error(f"Duplicate check error: {e}")
            return []
```

```python
# utils/progress_tracker.py
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import redis
import json

from ..config.settings import settings

logger = logging.getLogger(__name__)

@dataclass
class ProgressInfo:
    total: int
    completed: int
    failed: int
    start_time: float
    current_file: str = ""
    status: str = "running"
    
    @property
    def percentage(self) -> float:
        return (self.completed + self.failed) / self.total * 100 if self.total > 0 else 0
    
    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time
    
    @property
    def eta(self) -> Optional[float]:
        if self.completed == 0:
            return None
        
        avg_time_per_item = self.elapsed_time / self.completed
        remaining_items = self.total - self.completed - self.failed
        return avg_time_per_item * remaining_items if remaining_items > 0 else 0

class ProgressTracker:
    """Track progress cho batch operations"""
    
    def __init__(self, total: int, job_id: str = None):
        self.total = total
        self.job_id = job_id or f"job_{int(time.time())}"
        
        self.progress = ProgressInfo(
            total=total,
            completed=0,
            failed=0,
            start_time=time.time()
        )
        
        # Redis connection for sharing progress
        try:
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                decode_responses=True
            )
            self.use_redis = True
        except Exception as e:
            logger.warning(f"Redis not available for progress tracking: {e}")
            self.use_redis = False
    
    def update(self, completed: int = 1, failed: int = 0, current_file: str = ""):
        """Update progress counters"""
        self.progress.completed += completed
        self.progress.failed += failed
        if current_file:
            self.progress.current_file = current_file
        
        # Update Redis
        if self.use_redis:
            try:
                progress_data = {
                    'total': self.progress.total,
                    'completed': self.progress.completed,
                    'failed': self.progress.failed,
                    'percentage': self.progress.percentage,
                    'elapsed_time': self.progress.elapsed_time,
                    'eta': self.progress.eta,
                    'current_file': self.progress.current_file,
                    'status': self.progress.status
                }
                
                self.redis_client.setex(
                    f"progress:{self.job_id}",
                    300,  # 5 minutes TTL
                    json.dumps(progress_data)
                )
            except Exception as e:
                logger.warning(f"Failed to update progress in Redis: {e}")
    
    def mark_completed(self):
        """Mark job as completed"""
        self.progress.status = "completed"
        self.update()
        
        logger.info(f"Job {self.job_id} completed: {self.progress.completed} successful, {self.progress.failed} failed")
    
    def mark_failed(self, error_msg: str = ""):
        """Mark job as failed"""
        self.progress.status = "failed"
        if error_msg:
            logger.error(f"Job {self.job_id} failed: {error_msg}")
        self.update()
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current progress info"""
        return {
            'job_id': self.job_id,
            'total': self.progress.total,
            'completed': self.progress.completed,
            'failed': self.progress.failed,
            'percentage': self.progress.percentage,
            'elapsed_time': self.progress.elapsed_time,
            'eta': self.progress.eta,
            'current_file': self.progress.current_file,
            'status': self.progress.status
        }
    
    def print_progress(self):
        """Print progress to console"""
        percentage = self.progress.percentage
        elapsed = self.progress.elapsed_time
        eta = self.progress.eta or 0
        
        status_line = (
            f"Progress: {self.progress.completed + self.progress.failed}/{self.progress.total} "
            f"({percentage:.1f}%) | "
            f"Success: {self.progress.completed} | "
            f"Failed: {self.progress.failed} | "
            f"Elapsed: {elapsed:.1f}s | "
            f"ETA: {eta:.1f}s"
        )
        
        if self.progress.current_file:
            status_line += f" | Current: {self.progress.current_file}"
        
        print(f"\r{status_line}", end="", flush=True)
```

## 10. File Handler

```python
# utils/file_handler.py
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import fnmatch

logger = logging.getLogger(__name__)

class FileHandler:
    """Handle file operations cho document ingestion"""
    
    def __init__(self):
        self.supported_extensions = ['.pdf', '.docx', '.md', '.txt']
    
    def find_documents(self, directory: str, patterns: List[str] = None) -> List[str]:
        """Find tất cả documents trong directory theo patterns"""
        try:
            directory = Path(directory)
            if not directory.exists():
                logger.error(f"Directory does not exist: {directory}")
                return []
            
            if not directory.is_dir():
                logger.error(f"Path is not a directory: {directory}")
                return []
            
            patterns = patterns or ['*.pdf', '*.docx', '*.md', '*.txt']
            found_files = []
            
            for pattern in patterns:
                # Find files matching pattern
                for file_path in directory.rglob(pattern):
                    if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                        found_files.append(str(file_path))
            
            # Remove duplicates và sort
            found_files = sorted(list(set(found_files)))
            
            logger.info(f"Found {len(found_files)} documents in {directory}")
            return found_files
            
        except Exception as e:
            logger.error(f"Error finding documents: {e}")
            return []
    
    def organize_files_by_type(self, file_paths: List[str]) -> Dict[str, List[str]]:
        """Organize files by their extensions"""
        organized = {}
        
        for file_path in file_paths:
            extension = Path(file_path).suffix.lower()
            if extension not in organized:
                organized[extension] = []
            organized[extension].append(file_path)
        
        return organized
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """Get comprehensive file information"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return {"error": "File does not exist"}
            
            stat = file_path.stat()
            
            return {
                'name': file_path.name,
                'stem': file_path.stem,
                'extension': file_path.suffix,
                'size': stat.st_size,
                'size_mb': stat.st_size / (1024 * 1024),
                'created': stat.st_ctime,
                'modified': stat.st_mtime,
                'path': str(file_path),
                'parent_dir': str(file_path.parent)
            }
            
        except Exception as e:
            logger.error(f"Error getting file info for {file_path}: {e}")
            return {"error": str(e)}
    
    def create_backup(self, file_path: str, backup_dir: str = None) -> str:
        """Create backup của file trước khi processing"""
        try:
            source_path = Path(file_path)
            
            if not backup_dir:
                backup_dir = source_path.parent / "backups"
            else:
                backup_dir = Path(backup_dir)
            
            backup_dir.mkdir(exist_ok=True)
            
            # Create backup filename với timestamp
            import time
            timestamp = int(time.time())
            backup_filename = f"{source_path.stem}_{timestamp}{source_path.suffix}"
            backup_path = backup_dir / backup_filename
            
            # Copy file
            import shutil
            shutil.copy2(source_path, backup_path)
            
            logger.info(f"Created backup: {backup_path}")
            return str(backup_path)
            
        except Exception as e:
            logger.error(f"Failed to create backup for {file_path}: {e}")
            raise
    
    def validate_directory_structure(self, base_dir: str) -> Dict[str, Any]:
        """Validate directory structure cho batch processing"""
        try:
            base_path = Path(base_dir)
            
            validation_result = {
                'valid': True,
                'total_files': 0,
                'supported_files': 0,
                'unsupported_files': 0,
                'empty_files': 0,
                'large_files': 0,
                'file_types': {},
                'issues': []
            }
            
            max_file_size = 50 * 1024 * 1024  # 50MB
            
            for file_path in base_path.rglob('*'):
                if file_path.is_file():
                    validation_result['total_files'] += 1
                    
                    extension = file_path.suffix.lower()
                    if extension not in validation_result['file_types']:
                        validation_result['file_types'][extension] = 0
                    validation_result['file_types'][extension] += 1
                    
                    # Check if supported
                    if extension in self.supported_extensions:
                        validation_result['supported_files'] += 1
                    else:
                        validation_result['unsupported_files'] += 1
                        validation_result['issues'].append(f"Unsupported file type: {file_path}")
                    
                    # Check file size
                    try:
                        file_size = file_path.stat().st_size
                        if file_size == 0:
                            validation_result['empty_files'] += 1
                            validation_result['issues'].append(f"Empty file: {file_path}")
                        elif file_size > max_file_size:
                            validation_result['large_files'] += 1
                            validation_result['issues'].append(f"Large file ({file_size} bytes): {file_path}")
                    except Exception as e:
                        validation_result['issues'].append(f"Cannot access file {file_path}: {e}")
            
            if validation_result['issues']:
                validation_result['valid'] = False
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Directory validation error: {e}")
            return {
                'valid': False,
                'error': str(e)
            }
```

## 11. Setup Script

```python
# setup.py
#!/usr/bin/env python3
"""
Setup script cho Data Ingestion Tool
Cài đặt dependencies và initialize system
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check Python version requirement"""
    if sys.version_info < (3, 8):
        print("Python 3.8+ is required")
        sys.exit(1)
    print(f"Python version: {sys.version}")

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)

def create_directories():
    """Create necessary directories"""
    directories = [
        "faiss_collections",
        "logs",
        "temp",
        "backups",
        "tests/sample_documents"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {dir_path}")

def check_external_dependencies():
    """Check external dependencies"""
    dependencies = {
        'redis-server': 'Redis server',
        'psql': 'PostgreSQL client'
    }
    
    missing = []
    for cmd, desc in dependencies.items():
        if not shutil.which(cmd):
            missing.append(f"{cmd} ({desc})")
    
    if missing:
        print("Warning: Missing external dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("Please install them manually")

def create_env_template():
    """Create .env template if not exists"""
    env_template = """
# Database Configuration
DATABASE_URL=postgresql://username:password@localhost:5432/chatbot_db

# Redis Configuration  
REDIS_HOST=localhost
REDIS_PORT=6379

# FAISS Configuration
FAISS_COLLECTIONS_PATH=./faiss_collections
EMBEDDING_MODEL=keepitreal/vietnamese-sbert
EMBEDDING_DIMENSION=768

# Processing Configuration
MAX_WORKERS=4
BATCH_SIZE=50

# Quality Control
MIN_WORD_COUNT=50
MAX_WORD_COUNT=1000
LANGUAGE_CONFIDENCE_THRESHOLD=0.8
DUPLICATE_THRESHOLD=0.95
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        env_file.write_text(env_template.strip())
        print("Created .env template file - please update with your configuration")
    else:
        print(".env file already exists")

def main():
    """Main setup function"""
    print("Setting up Data Ingestion Tool...")
    
    check_python_version()
    install_dependencies()
    create_directories()
    check_external_dependencies()
    create_env_template()
    
    print("\nSetup completed!")
    print("\nNext steps:")
    print("1. Update .env file with your configuration")
    print("2. Ensure PostgreSQL and Redis are running")
    print("3. Run: python main.py --help")

if __name__ == "__main__":
    main()
```

## Cách sử dụng tool

```bash
# 1. Setup initial
python setup.py

# 2. Ingest single document
python main
Tiếp tục hướng dẫn sử dụng và các file còn thiếu:

## 12. Web Interface (Optional)

```python
# web/app.py
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
import os
import tempfile
from typing import List, Optional
import uuid

from ..core.document_parser import DocumentParser
from ..core.content_processor import ContentProcessor
from ..core.metadata_engine import MetadataEngine
from ..core.embedding_generator import EmbeddingGenerator
from ..core.collection_manager import CollectionManager
from ..utils.progress_tracker import ProgressTracker
from ..utils.validation import DataValidator

app = FastAPI(title="Data Ingestion Web Interface", version="1.0.0")
templates = Jinja2Templates(directory="web/templates")

# Background jobs tracking
active_jobs = {}

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard"""
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    collection: str = Form(...),
    category: Optional[str] = Form(None)
):
    """Upload và process single document"""
    
    # Validate file
    if not file.filename.endswith(('.pdf', '.docx', '.md', '.txt')):
        raise HTTPException(status_code=400, detail="Unsupported file type")
    
    # Create temp file
    job_id = str(uuid.uuid4())
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1])
    
    try:
        content = await file.read()
        temp_file.write(content)
        temp_file.flush()
        
        # Start background processing
        background_tasks.add_task(
            process_document_background,
            job_id, temp_file.name, collection, category, file.filename
        )
        
        return {"job_id": job_id, "status": "started", "filename": file.filename}
        
    except Exception as e:
        os.unlink(temp_file.name)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        temp_file.close()

@app.get("/job/{job_id}/status")
async def get_job_status(job_id: str):
    """Get processing job status"""
    if job_id in active_jobs:
        tracker = active_jobs[job_id]
        return tracker.get_progress()
    else:
        return {"error": "Job not found"}

@app.get("/collections")
async def list_collections():
    """API endpoint để list collections"""
    from ..config.settings import settings
    collections_config = settings.load_collections_config()['collections']
    
    collections_info = []
    for name, config in collections_config.items():
        collections_info.append({
            'name': name,
            'description': config.get('description', ''),
            'categories': config.get('categories', [])
        })
    
    return collections_info

async def process_document_background(job_id: str, file_path: str, collection: str, 
                                    category: Optional[str], original_filename: str):
    """Background task để process document"""
    tracker = ProgressTracker(total=1, job_id=job_id)
    active_jobs[job_id] = tracker
    
    try:
        # Initialize components
        parser = DocumentParser()
        processor = ContentProcessor()
        metadata_engine = MetadataEngine()
        embedding_generator = EmbeddingGenerator()
        collection_manager = CollectionManager()
        validator = DataValidator()
        
        tracker.update(current_file=original_filename)
        
        # Process document
        parsed_doc = parser.parse_document(file_path)
        chunks = processor.process_content(parsed_doc)
        
        document_chunks = []
        for i, chunk_content in enumerate(chunks):
            metadata = await metadata_engine.generate_metadata(
                content=chunk_content,
                title=parsed_doc.title,
                collection=collection,
                category=category,
                chunk_index=i,
                chunk_total=len(chunks),
                source_file=original_filename
            )
            
            if validator.validate_chunk_quality(chunk_content, metadata):
                embedding = await embedding_generator.generate_embedding(chunk_content)
                
                from ..models.document import DocumentChunk
                doc_chunk = DocumentChunk(
                    content=chunk_content,
                    metadata=metadata,
                    embedding=embedding
                )
                document_chunks.append(doc_chunk)
        
        # Save to collection
        added_count = await collection_manager.add_documents_to_collection(
            collection, document_chunks
        )
        
        tracker.update(completed=1)
        tracker.mark_completed()
        
    except Exception as e:
        tracker.mark_failed(str(e))
    finally:
        # Cleanup temp file
        try:
            os.unlink(file_path)
        except:
            pass
        
        # Remove from active jobs after some time
        import asyncio
        await asyncio.sleep(300)  # 5 minutes
        if job_id in active_jobs:
            del active_jobs[job_id]
```

```html
<!-- web/templates/dashboard.html -->
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Ingestion Dashboard</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; }
        .upload-section { background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .progress { background: #e0e0e0; border-radius: 4px; height: 20px; margin: 10px 0; }
        .progress-bar { background: #4CAF50; height: 100%; border-radius: 4px; transition: width 0.3s; }
        .job-status { background: #fff; padding: 15px; border: 1px solid #ddd; margin: 10px 0; border-radius: 4px; }
        select, input, button { padding: 8px 12px; margin: 5px; border: 1px solid #ddd; border-radius: 4px; }
        button { background: #007bff; color: white; cursor: pointer; }
        button:hover { background: #0056b3; }
        .error { color: #dc3545; }
        .success { color: #28a745; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Data Ingestion Tool Dashboard</h1>
        
        <div class="upload-section">
            <h2>Upload Document</h2>
            <form id="uploadForm" enctype="multipart/form-data">
                <div>
                    <label for="file">Select File:</label>
                    <input type="file" id="file" name="file" accept=".pdf,.docx,.md,.txt" required>
                </div>
                
                <div>
                    <label for="collection">Target Collection:</label>
                    <select id="collection" name="collection" required>
                        <option value="">Select Collection</option>
                    </select>
                </div>
                
                <div>
                    <label for="category">Category (Optional):</label>
                    <input type="text" id="category" name="category" placeholder="e.g., security, pricing">
                </div>
                
                <button type="submit">Upload and Process</button>
            </form>
        </div>
        
        <div id="jobsSection">
            <h2>Processing Jobs</h2>
            <div id="jobsList"></div>
        </div>
    </div>
    
    <script>
        // Load collections on page load
        document.addEventListener('DOMContentLoaded', function() {
            loadCollections();
            checkActiveJobs();
        });
        
        async function loadCollections() {
            try {
                const response = await fetch('/collections');
                const collections = await response.json();
                
                const select = document.getElementById('collection');
                select.innerHTML = '<option value="">Select Collection</option>';
                
                collections.forEach(collection => {
                    const option = document.createElement('option');
                    option.value = collection.name;
                    option.textContent = `${collection.name} - ${collection.description}`;
                    select.appendChild(option);
                });
            } catch (error) {
                console.error('Failed to load collections:', error);
            }
        }
        
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(this);
            const button = this.querySelector('button');
            button.disabled = true;
            button.textContent = 'Processing...';
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                
                if (response.ok) {
                    addJobToList(result.job_id, result.filename);
                    this.reset();
                } else {
                    alert('Error: ' + result.detail);
                }
            } catch (error) {
                alert('Upload failed: ' + error.message);
            } finally {
                button.disabled = false;
                button.textContent = 'Upload and Process';
            }
        });
        
        function addJobToList(jobId, filename) {
            const jobsList = document.getElementById('jobsList');
            
            const jobElement = document.createElement('div');
            jobElement.className = 'job-status';
            jobElement.id = `job-${jobId}`;
            jobElement.innerHTML = `
                <h3>${filename}</h3>
                <div class="progress">
                    <div class="progress-bar" style="width: 0%"></div>
                </div>
                <div class="status">Status: Starting...</div>
                <div class="details"></div>
            `;
            
            jobsList.insertBefore(jobElement, jobsList.firstChild);
            
            // Start monitoring job
            monitorJob(jobId);
        }
        
        async function monitorJob(jobId) {
            const jobElement = document.getElementById(`job-${jobId}`);
            if (!jobElement) return;
            
            const progressBar = jobElement.querySelector('.progress-bar');
            const statusDiv = jobElement.querySelector('.status');
            const detailsDiv = jobElement.querySelector('.details');
            
            try {
                const response = await fetch(`/job/${jobId}/status`);
                const status = await response.json();
                
                if (status.error) {
                    statusDiv.textContent = 'Status: Error - Job not found';
                    statusDiv.className = 'status error';
                    return;
                }
                
                // Update progress
                progressBar.style.width = `${status.percentage}%`;
                statusDiv.textContent = `Status: ${status.status} (${status.percentage.toFixed(1)}%)`;
                
                // Update details
                detailsDiv.innerHTML = `
                    <small>
                        Completed: ${status.completed}/${status.total} | 
                        Failed: ${status.failed} | 
                        Elapsed: ${status.elapsed_time.toFixed(1)}s
                        ${status.eta ? ` | ETA: ${status.eta.toFixed(1)}s` : ''}
                    </small>
                `;
                
                // Set status color
                if (status.status === 'completed') {
                    statusDiv.className = 'status success';
                    progressBar.style.background = '#28a745';
                } else if (status.status === 'failed') {
                    statusDiv.className = 'status error';
                    progressBar.style.background = '#dc3545';
                } else {
                    // Continue monitoring
                    setTimeout(() => monitorJob(jobId), 2000);
                }
                
            } catch (error) {
                statusDiv.textContent = 'Status: Error monitoring job';
                statusDiv.className = 'status error';
            }
        }
        
        function checkActiveJobs() {
            // This would check for any active jobs from localStorage or server
            // Implementation depends on requirements
        }
    </script>
</body>
</html>
```

## 13. Test Suite

```python
# tests/test_document_parser.py
import pytest
import tempfile
import os
from pathlib import Path

from core.document_parser import DocumentParser
from models.document import DocumentType

@pytest.fixture
def parser():
    return DocumentParser()

@pytest.fixture
def sample_text_file():
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write("Đây là nội dung test tiếng Việt.\nDòng thứ hai của document.")
        temp_path = f.name
    
    yield temp_path
    os.unlink(temp_path)

def test_parse_text_document(parser, sample_text_file):
    """Test parsing text document"""
    parsed_doc = parser.parse_document(sample_text_file)
    
    assert parsed_doc.document_type == DocumentType.TEXT
    assert "Đây là nội dung test" in parsed_doc.content
    assert parsed_doc.title is not None
    assert parsed_doc.metadata['language'] == 'vi'

def test_unsupported_file_type(parser):
    """Test unsupported file type"""
    with tempfile.NamedTemporaryFile(suffix='.xyz') as f:
        with pytest.raises(ValueError):
            parser.parse_document(f.name)

def test_nonexistent_file(parser):
    """Test nonexistent file"""
    with pytest.raises(FileNotFoundError):
        parser.parse_document("/path/that/does/not/exist.txt")
```

```python
# tests/test_content_processor.py
import pytest
from core.content_processor import ContentProcessor

@pytest.fixture
def processor():
    return ContentProcessor()

def test_vietnamese_keyword_extraction(processor):
    """Test Vietnamese keyword extraction"""
    content = "Sản phẩm A có tính năng bảo mật SSL và authentication mạnh mẽ. API integration rất dễ sử dụng."
    
    keywords = processor.extract_keywords(content, top_k=5)
    
    assert len(keywords) <= 5
    assert any('sản phẩm' in kw.lower() or 'bảo mật' in kw.lower() for kw in keywords)

def test_content_chunking(processor):
    """Test smart content chunking"""
    long_content = "Đây là câu đầu tiên. " * 100  # Tạo content dài
    
    chunks = processor.process_content(
        type('MockDoc', (), {
            'content': long_content,
            'title': 'Test Doc'
        })(),
        max_chunk_size=50,
        overlap_size=10
    )
    
    assert len(chunks) > 1
    assert all(len(chunk.split()) <= 60 for chunk in chunks)  # Allow some overhead

def test_chunk_validation(processor):
    """Test chunk quality validation"""
    good_chunk = "Đây là một đoạn text có chất lượng tốt với nhiều từ có ý nghĩa và cấu trúc rõ ràng."
    bad_chunk = "abc 123 !!!"
    
    assert processor._validate_chunk(good_chunk) == True
    assert processor._validate_chunk(bad_chunk) == False
```

## 14. Cách sử dụng hoàn chỉnh

```bash
# 1. Setup môi trường
git clone <repo>
cd data_ingestion_tool
python setup.py

# 2. Cập nhật .env file với config thực tế
vim .env

# 3. Khởi động services
docker-compose up -d redis postgres

# 4. Test connection
python -c "
from config.settings import settings
import redis
import psycopg2

# Test Redis
r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
print('Redis:', r.ping())

# Test PostgreSQL  
conn = psycopg2.connect(settings.DATABASE_URL)
print('PostgreSQL: Connected')
conn.close()
"

# 5. Ingest documents

# Single file
python main.py ingest --file documents/product_guide.pdf --collection product_a_features --category security

# Batch processing
python main.py batch-ingest --dir ./documents --auto-route

# Với collection cụ thể
python main.py batch-ingest --dir ./documents --collection product_a_features

# 6. Validation trước khi ingest
python main.py validate --file documents/test.pdf --dry-run

# 7. Quản lý collections
python main.py list-collections
python main.py collection-stats --name product_a_features
python main.py quality-check --collection all --min-score 0.7

# 8. Monitoring
python main.py logs --tail 100

# 9. Web interface (optional)
cd web
uvicorn app:app --host 0.0.0.0 --port 8001
```

## 15. Docker Support

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements và install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p faiss_collections logs temp backups

EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from config.settings import settings; import redis; r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT); r.ping()" || exit 1

CMD ["python", "main.py", "--help"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  data-ingestion:
    build: .
    volumes:
      - ./faiss_collections:/app/faiss_collections
      - ./documents:/app/documents
      - ./logs:/app/logs
    environment:
      - DATABASE_URL=postgresql://postgres:password@postgres:5432/chatbot_db
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    depends_on:
      - postgres
      - redis
    ports:
      - "8001:8001"

  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: chatbot_db
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

Tool này cung cấp một hệ thống quản lý database hoàn chỉnh với:

- **Hỗ trợ tiếng Việt**: Sử dụng pyvi cho xử lý ngôn ngữ
- **Multiple formats**: PDF, DOCX, Markdown, Text
- **Quality control**: Comprehensive validation
- **CLI và Web interface**: Linh hoạt sử dụng
- **Batch processing**: Xử lý hàng loạt documents
- **Monitoring**: Progress tracking và health checks
- **Vietnamese-optimized**: Embedding models và keyword extraction cho tiếng Việt
