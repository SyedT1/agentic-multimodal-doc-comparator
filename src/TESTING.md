# Testing Guide for Document Comparison App

## Quick Start Testing

### 1. Install Dependencies (if not done)

```bash
pip install -r requirements.txt
```

**Expected time**: 5-10 minutes (large packages like PyTorch)

### 2. Create Test Documents

```bash
python create_test_docs.py
```

This creates three test documents:
- `test_doc1.docx` - Product requirements document
- `test_doc2.docx` - Similar document with differences
- `test_doc3_identical.docx` - Identical to doc1

### 3. Run the App

```bash
streamlit run streamlit_app.py
```

The app will open at: `http://localhost:8501`

### 4. Test Scenarios

#### Test Case 1: Similar Documents (Expected: 60-80% similarity)
- **Document 1**: test_doc1.docx
- **Document 2**: test_doc2.docx
- **What to expect**:
  - Overall similarity: ~65-75%
  - Text similarity: ~70-80% (similar topics, some wording differences)
  - Table similarity: ~50-60% (different tech stacks)
  - Matched sections showing overlapping features and overview

#### Test Case 2: Identical Documents (Expected: ~100% similarity)
- **Document 1**: test_doc1.docx
- **Document 2**: test_doc3_identical.docx
- **What to expect**:
  - Overall similarity: ~95-100%
  - Text similarity: ~100%
  - Table similarity: ~100%
  - All sections matched

#### Test Case 3: Test with Your Own Documents
- Upload any two PDF or DOCX files (max 50MB each)
- Adjust text/table weights in sidebar
- View detailed comparison results

## What to Look For

### ✅ Successful Run Indicators

1. **Progress bar completes** through all stages:
   - Ingesting documents
   - Extracting and embedding text
   - Extracting and embedding tables
   - Comparing documents

2. **Results display shows**:
   - Overall similarity gauge (0-100%)
   - Bar chart with text and table scores
   - Matched sections with content snippets
   - Page numbers for each match

3. **Download button** works and exports JSON report

### ⚠️ Common Issues to Check

1. **"Module not found" errors**
   - Run: `pip install -r requirements.txt`

2. **Model download on first run**
   - sentence-transformers will download ~90MB model first time
   - This is normal and only happens once

3. **Memory warnings**
   - Test with smaller documents first
   - Close other applications if needed

4. **Table extraction issues**
   - Some PDFs may have tables in image format (won't extract)
   - DOCX tables extract more reliably

## Expected Performance

- **Small documents** (< 5 pages): 5-15 seconds
- **Medium documents** (5-20 pages): 15-45 seconds
- **Large documents** (> 20 pages): 45+ seconds

## Verifying Results

### Text Similarity
- Check "Matched Sections" to see side-by-side text comparisons
- Higher scores = more semantic overlap
- Look for similar topics even with different wording

### Table Similarity
- Compares table schemas (headers) and content
- Identical tables = high score
- Different schemas = lower score

### Overall Score
- Weighted combination (default: 60% text, 40% table)
- Adjust weights in sidebar to change emphasis

## Troubleshooting

### App won't start
```bash
# Check Python version (need 3.8+)
python --version

# Reinstall streamlit
pip install --upgrade streamlit
```

### Embeddings slow
- First run downloads model (~90MB)
- Subsequent runs use cached model
- Consider using GPU if available (change to faiss-gpu in requirements)

### No matches found
- Documents may be too different
- Try adjusting chunk size in config.py
- Check if documents have extractable text (not scanned images)

## Advanced Testing

### Modify Configuration
Edit `config.py` to adjust:
```python
TEXT_CHUNK_SIZE = 512  # Increase for longer context
TEXT_CHUNK_OVERLAP = 50  # Increase for better matching
MODALITY_WEIGHTS = {"text": 0.60, "table": 0.40}  # Adjust importance
```

### Test Different Document Types
1. **Highly similar**: Same document, minor edits
2. **Moderately similar**: Same topic, different authors
3. **Dissimilar**: Completely different topics

### Validate Accuracy
Compare app results with manual review:
- Do matched sections make sense?
- Are similarity percentages reasonable?
- Are table comparisons accurate?

## Next Steps

After successful testing:
1. Test with your real documents
2. Adjust weights based on your use case
3. Consider Phase 2 features (image, layout, metadata comparison)
4. Provide feedback for improvements

## Support

If you encounter issues:
1. Check error message in terminal
2. Verify all dependencies installed
3. Ensure documents are valid PDF/DOCX
4. Check file size limits (50MB default)
