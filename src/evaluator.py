from difflib import SequenceMatcher


class OCREvaluator:
    """
    Evaluates OCR accuracy by comparing with ground truth.
    """
    
    def __init__(self):
        print("Evaluator ready")

    # Text similarity
    def calculate_similarity(self, text1, text2):
        """
        Calculate similarity ratio between two texts
        
        Args:
            text1: First text string
            text2: Second text string
        
        Returns:
            Similarity ratio (0.0 to 1.0)
        """
        # Normalize texts (lowercase, strip whitespace)
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()
        
        return SequenceMatcher(None, t1, t2).ratio()

    # Compare OCR with ground truth
    def evaluate_extraction(self, ocr_result, ground_truth_boxes):
        """
        Compare OCR output with ground truth
        
        Args:
            ocr_result: Output from ConfidenceOCR
            ground_truth_boxes: Ground truth from data_loader
        
        Returns:
            dict with evaluation metrics
        """
        # Combine OCR words into full text
        ocr_text = ' '.join([w['text'] for w in ocr_result['words']])
        
        # Combine ground truth into full text
        gt_text = ' '.join([box['text'] for box in ground_truth_boxes])
        
        # Calculate similarity
        similarity = self.calculate_similarity(ocr_text, gt_text)
        
        return {
            'ocr_text': ocr_text,
            'ground_truth': gt_text,
            'similarity': similarity,
            'accuracy_percentage': similarity * 100,
            'ocr_word_count': len(ocr_result['words']),
            'gt_word_count': len(ground_truth_boxes)
        }

    # Check key field extraction

    def evaluate_key_fields(self, ocr_result, entities):
        """
        Check if key fields (company, date, total, address) were extracted
        
        Args:
            ocr_result: Output from ConfidenceOCR
            entities: Ground truth entities
        
        Returns:
            dict with field-level results
        """
        ocr_text = ocr_result['full_text'].lower()
        
        results = {}
        for field, value in entities.items():
            # Check if value appears in OCR text
            found = value.lower() in ocr_text
            
            # Calculate similarity for partial matches
            similarity = max([
                self.calculate_similarity(value, word['text'])
                for word in ocr_result['words']
            ]) if ocr_result['words'] else 0
            
            results[field] = {
                'value': value,
                'found': found,
                'similarity': similarity
            }
        
        # Calculate overall field accuracy
        found_count = sum(1 for r in results.values() if r['found'])
        field_accuracy = (found_count / len(results) * 100) if results else 0
        
        return {
            'fields': results,
            'found_count': found_count,
            'total_fields': len(results),
            'field_accuracy': field_accuracy
        }

    # Analyze confidence vs accuracy
    def analyze_confidence_correlation(self, ocr_result, analysis, ground_truth_boxes):
        """
        Check if low confidence words are actually mistakes
        
        Args:
            ocr_result: OCR output
            analysis: Confidence analysis
            ground_truth_boxes: Ground truth
        
        Returns:
            dict with correlation metrics
        """
        gt_text_lower = ' '.join([box['text'].lower() for box in ground_truth_boxes])
        
        # Check low confidence words
        low_conf_words = analysis['low_confidence']['words']
        low_conf_correct = 0
        
        for word in low_conf_words:
            if word['text'].lower() in gt_text_lower:
                low_conf_correct += 1
        
        # Check high confidence words
        high_conf_words = analysis['high_confidence']['words']
        high_conf_correct = 0
        
        for word in high_conf_words:
            if word['text'].lower() in gt_text_lower:
                high_conf_correct += 1
        
        return {
            'low_confidence': {
                'total': len(low_conf_words),
                'correct': low_conf_correct,
                'accuracy': (low_conf_correct / len(low_conf_words) * 100) if low_conf_words else 0
            },
            'high_confidence': {
                'total': len(high_conf_words),
                'correct': high_conf_correct,
                'accuracy': (high_conf_correct / len(high_conf_words) * 100) if high_conf_words else 0
            }
        }
    # Generate evaluation report
    def generate_report(self, sample_id, ocr_result, analysis, ground_truth_boxes, entities):
        """
        Generate complete evaluation report
        
        Args:
            sample_id: Receipt ID
            ocr_result: OCR output
            analysis: Confidence analysis
            ground_truth_boxes: Ground truth boxes
            entities: Ground truth entities
        
        Returns:
            Complete evaluation report dict
        """
        extraction_eval = self.evaluate_extraction(ocr_result, ground_truth_boxes)
        field_eval = self.evaluate_key_fields(ocr_result, entities)
        correlation = self.analyze_confidence_correlation(ocr_result, analysis, ground_truth_boxes)
        
        return {
            'sample_id': sample_id,
            'text_similarity': extraction_eval['accuracy_percentage'],
            'field_accuracy': field_eval['field_accuracy'],
            'avg_confidence': ocr_result['avg_confidence'],
            'words_needing_review': analysis['low_confidence']['count'],
            'review_percentage': analysis['statistics']['review_percentage'],
            'low_conf_accuracy': correlation['low_confidence']['accuracy'],
            'high_conf_accuracy': correlation['high_confidence']['accuracy'],
            'full_report': {
                'extraction': extraction_eval,
                'fields': field_eval,
                'confidence_correlation': correlation
            }
        }

# Test code
if __name__ == "__main__":
    from data_loader import SROIEDataLoader
    from ocr_engine import ConfidenceOCR
    from confidence_scorer import ConfidenceAnalyzer
    
    print("=" * 60)
    print(" Testing Evaluator")
    print("=" * 60)
    
    # Load sample
    loader = SROIEDataLoader("data/raw/SROIE2019")
    samples = loader.get_sample_ids("train")
    sample = loader.load_complete_sample(samples[0])
    
    print(f" Testing on: {sample['sample_id']}")
    
    # Run OCR
    ocr = ConfidenceOCR()
    ocr_result = ocr.extract_text_with_confidence(sample['image'])
    
    # Analyze
    analyzer = ConfidenceAnalyzer()
    analysis = analyzer.analyze(ocr_result)
    
    # Evaluate
    evaluator = OCREvaluator()
    report = evaluator.generate_report(
        sample['sample_id'],
        ocr_result,
        analysis,
        sample['boxes'],
        sample['entities']
    )
    
    # Print results
    print()
    print("=" * 60)
    print(" EVALUATION RESULTS")
    print("=" * 60)
    print(f"Sample ID: {report['sample_id']}")
    print(f"Text Similarity: {report['text_similarity']:.1f}%")
    print(f"Field Accuracy: {report['field_accuracy']:.1f}%")
    print(f"Avg Confidence: {report['avg_confidence']:.1f}%")
    print()
    print(f"Words Needing Review: {report['words_needing_review']} ({report['review_percentage']:.1f}%)")
    print()
    print("Confidence vs Accuracy:")
    print(f"  Low confidence accuracy: {report['low_conf_accuracy']:.1f}%")
    print(f"  High confidence accuracy: {report['high_conf_accuracy']:.1f}%")
    print()
    print(" Test passed!")
    print("=" * 60)