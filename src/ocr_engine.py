import pytesseract
from PIL import Image
import numpy as np

# Configure Tesseract path 
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class ConfidenceOCR:
    """
    OCR engine that extracts text with word-level confidence scores.
    """
    
    def __init__(self, min_confidence=0):
        """
        Initialize OCR engine
        
        Args:
            min_confidence: Minimum confidence to consider (0-100)
        """
        self.min_confidence = min_confidence
        print(f" OCR engine ready (min confidence: {min_confidence}%)")

    # Main OCR function
    def extract_text_with_confidence(self, image):
        """
        Run OCR and get word-level confidence scores
        
        Args:
            image: PIL Image object
        
        Returns:
            dict with:
                - full_text: Complete extracted text
                - words: List of {text, confidence, bbox}
                - avg_confidence: Average confidence score
        """
        # Get detailed OCR data
        ocr_data = pytesseract.image_to_data(
            image,
            output_type=pytesseract.Output.DICT
        )
        
        # Extract words with confidence
        words = []
        for i in range(len(ocr_data['text'])):
            text = ocr_data['text'][i].strip()
            conf = int(ocr_data['conf'][i])
            
            # Filter by minimum confidence
            if conf >= self.min_confidence and text:
                bbox = [
                    ocr_data['left'][i],
                    ocr_data['top'][i],
                    ocr_data['left'][i] + ocr_data['width'][i],
                    ocr_data['top'][i] + ocr_data['height'][i]
                ]
                
                words.append({
                    'text': text,
                    'confidence': conf,
                    'bbox': bbox
                })
        
        # Get full text
        full_text = pytesseract.image_to_string(image)
        
        # Calculate average confidence
        confidences = [w['confidence'] for w in words]
        avg_conf = np.mean(confidences) if confidences else 0
        
        return {
            'full_text': full_text.strip(),
            'words': words,
            'avg_confidence': float(avg_conf),
            'total_words': len(words)
        }
    # Quick text extraction
    def extract_text_only(self, image):
        """
        Quick OCR without confidence scores
        
        Args:
            image: PIL Image object
        
        Returns:
            Extracted text as string
        """
        return pytesseract.image_to_string(image).strip()

# Test code
if __name__ == "__main__":
    from data_loader import SROIEDataLoader

    print(" Testing OCR Engine")
    # Load a sample receipt
    loader = SROIEDataLoader("data/raw/SROIE2019")
    samples = loader.get_sample_ids("train")
    sample = loader.load_complete_sample(samples[0])
    
    print(f" Testing on: {sample['sample_id']}")
    
    # Run OCR
    ocr = ConfidenceOCR(min_confidence=0)
    result = ocr.extract_text_with_confidence(sample['image'])
    
    # Show results
    print(f" Extracted {result['total_words']} words")
    print(f" Average confidence: {result['avg_confidence']:.1f}%")
    print()
    
    # Show first 5 words
    print(" First 5 words:")
    for word in result['words'][:5]:
        print(f"   '{word['text']:<15}' → {word['confidence']:>3}%")
    
    print("\n Test passed!")
    print("=" * 60)