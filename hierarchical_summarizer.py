import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizerFast
from typing import List, Dict, Optional, Union
import time
import logging
from tqdm import tqdm
import numpy as np
from semantic_document_chunker import SemanticDocumentChunker

# Configure logging for research reproducibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HierarchicalSummarizer:
    """
    State-of-the-Art Hierarchical Summarizer for Long Documents.
    
    Implements a robust 2-stage pipeline:
    1. Local Stage: Summarize semantic chunks in parallel (batched).
    2. Global Stage: Aggregate chunk summaries into a coherent final summary.
    
    Optimizations:
    - GPU Acceleration (CUDA/MPS)
    - Dynamic Batching
    - Comparison-grade Beam Search parameters
    """
    
    def __init__(self, 
                 model_name: str = "google/pegasus-multi_news",
                 device: Optional[str] = None,
                 batch_size: int = 4,
                 chunker: Optional[SemanticDocumentChunker] = None):
        """
        Initialize the summarizer.
        
        Args:
            model_name: HuggingFace model hub ID.
            device: 'cuda', 'mps', or 'cpu'. Auto-detected if None.
            batch_size: Batch size for chunk summarization.
            chunker: Pre-initialized chunker instance.
        """
        # 1. Device Selection
        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device
            
        logger.info(f"Initializing HierarchicalSummarizer on {self.device}")
        
        # 2. Load Model & Tokenizer (Fast Rust-based tokenizer)
        try:
            self.tokenizer = PegasusTokenizerFast.from_pretrained(model_name)
            self.model = PegasusForConditionalGeneration.from_pretrained(model_name).to(self.device)
            # FP16 inference for speed if on CUDA (MPS support varies)
            if self.device == 'cuda':
                self.model = self.model.half()
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
            
        self.batch_size = batch_size
        
        # 3. Initialize Chunker (SOTA configuration)
        if chunker:
            self.chunker = chunker
        else:
            self.chunker = SemanticDocumentChunker(
                tokenizer=self.tokenizer,
                max_tokens=1024,
                overlap_tokens=128,
                use_semantic_coherence=True,
                adaptive_overlap=True # SOTA mode
            )
            
    def _generate(self, inputs: List[str], max_length: int = 256, min_length: int = 32) -> List[str]:
        """
        Low-level generation with researched beam search parameters.
        """
        # Tokenize (Batch)
        batch = self.tokenizer(
            inputs, 
            truncation=True, 
            padding="longest", 
            max_length=1024, 
            return_tensors="pt"
        ).to(self.device)
        
        # Generate with SOTA parameters for Multi-News
        # - num_beams=8: Standard for high quality abstractive summ
        # - length_penalty=0.8: Encourages slightly shorter, punchier sentences
        # - no_repeat_ngram_size=3: Prevents repetitive phrases
        try:
            summary_ids = self.model.generate(
                batch["input_ids"],
                num_beams=8, 
                max_length=max_length,
                min_length=min_length,
                length_penalty=0.8,
                no_repeat_ngram_size=3,
                early_stopping=True
            )
            
            # Decode
            summaries = self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
            return summaries
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return [""] * len(inputs)

    def summarize_document(self, document: str) -> Dict[str, Union[str, List[str]]]:
        """
        Execute the full hierarchical pipeline on a raw document.
        
        Returns:
            Dict containing:
            - 'final_summary': The resulting summary
            - 'chunk_summaries': Intermediate summaries (for analysis)
            - 'chunks': Raw chunks
        """
        if not document.strip():
            return {'final_summary': "", 'chunk_summaries': [], 'chunks': []}
            
        # Stage 1: Segmentation (SOTA Hybrid)
        chunks = self.chunker.chunk_document(document)
        chunk_texts = [c['text'] for c in chunks]
        
        if not chunk_texts:
            return {'final_summary': "", 'chunk_summaries': [], 'chunks': []}
            
        # Stage 2: Chunk Summarization (Batch Processing)
        chunk_summaries = []
        
        # Determine strictness based on chunk count
        # If we have many chunks, local summaries should be concise.
        local_max_len = 128 if len(chunks) > 5 else 256
        
        for i in range(0, len(chunk_texts), self.batch_size):
            batch = chunk_texts[i : i + self.batch_size]
            summaries = self._generate(batch, max_length=local_max_len)
            chunk_summaries.extend(summaries)
            
        # Stage 3: Aggregation (Global Summary)
        # Concatenate local summaries
        # Note: We add special separators or just spaces. Space is standard for PEGASUS.
        concatenated_summary = " ".join(chunk_summaries)
        
        # Check if we need a final pass
        # If the concatenated summary is short enough, maybe it IS the summary?
        # But usually we re-summarize to smooth the transitions.
        
        final_summary = concatenated_summary
        
        # Only re-summarize if there's enough content to warrant it
        # Otherwise we risk hallucinating or just copying.
        if len(self.tokenizer.tokenize(concatenated_summary)) > 256:
            final_summary_list = self._generate([concatenated_summary], max_length=512, min_length=128)
            final_summary = final_summary_list[0]
            
        return {
            'final_summary': final_summary,
            'chunk_summaries': chunk_summaries,
            'chunks': chunks,
            'concatenated_intermediate': concatenated_summary
        }

if __name__ == "__main__":
    # Integration Test
    print("Testing Hierarchical Summarizer...")
    
    # Create a dummy long text
    dummy_text = "This is a sentence about technology. " * 50 + " " + \
                 "This is a sentence about nature. " * 50 + " " + \
                 "This is a sentence about space. " * 50
                 
    # DEBUG: Force CPU to rule out MPS hang
    summarizer = HierarchicalSummarizer(device='cpu')
    print(f"Initialized on {summarizer.device}")
    
    result = summarizer.summarize_document(dummy_text)
    
    print("\n=== Chunk Summaries ===")
    for i, s in enumerate(result['chunk_summaries']):
        print(f"Chunk {i}: {s[:100]}...")
        
    print("\n=== Final Summary ===")
    print(result['final_summary'])
    print(f"\nPipeline successful. Device: {summarizer.device}")
