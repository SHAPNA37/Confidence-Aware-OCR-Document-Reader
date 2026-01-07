import json
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from datetime import datetime

from src.data_loader import SROIEDataLoader
from src.ocr_engine import ConfidenceOCR
from src.confidence_scorer import ConfidenceAnalyzer
from src.visualizer import ConfidenceVisualizer
from src.evaluator import OCREvaluator


class OCRPipeline:
    """
    Complete OCR pipeline with confidence analysis.
    """
    
    def __init__(self, data_dir, output_dir="data/results"):
        """
        Initialize pipeline
        
        Args:
            data_dir: Path to SROIE2019 folder
            output_dir: Where to save results
        """
        self.loader = SROIEDataLoader(data_dir)
        self.ocr = ConfidenceOCR()
        self.analyzer = ConfidenceAnalyzer()
        self.visualizer = ConfidenceVisualizer()
        self.evaluator = OCREvaluator()
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✅ Pipeline initialized")
        print(f"📁 Output directory: {self.output_dir.absolute()}")
    
    # ---------------------------
    # Process single receipt
    # ---------------------------
    def process_single(self, sample_id, split='train', save_visualizations=True):
        """
        Process one receipt through complete pipeline
        
        Args:
            sample_id: Receipt ID
            split: 'train' or 'test'
            save_visualizations: Whether to save annotated images
        
        Returns:
            dict with all results
        """
        # Load data
        sample = self.loader.load_complete_sample(sample_id, split)
        
        # Run OCR
        ocr_result = self.ocr.extract_text_with_confidence(sample['image'])
        
        # Analyze confidence
        analysis = self.analyzer.analyze(ocr_result)
        
        # Evaluate accuracy
        report = self.evaluator.generate_report(
            sample_id,
            ocr_result,
            analysis,
            sample['boxes'],
            sample['entities']
        )
        
        # Save visualizations
        if save_visualizations:
            vis_dir = self.output_dir / "visualizations"
            vis_dir.mkdir(exist_ok=True)
            
            self.visualizer.visualize_confidence(
                sample['image'],
                ocr_result,
                save_path=vis_dir / f"{sample_id}_confidence.jpg"
            )
            
            self.visualizer.highlight_review_areas(
                sample['image'],
                analysis,
                save_path=vis_dir / f"{sample_id}_review.jpg"
            )
        
        return report
    
    # ---------------------------
    # Process multiple receipts
    # ---------------------------
    def process_batch(self, num_samples=None, split='train', save_visualizations=False):
        """
        Process multiple receipts
        
        Args:
            num_samples: How many to process (None = all)
            split: 'train' or 'test'
            save_visualizations: Save images (slower)
        
        Returns:
            List of reports
        """
        # Get sample IDs
        sample_ids = self.loader.get_sample_ids(split)
        
        if num_samples:
            sample_ids = sample_ids[:num_samples]
        
        print(f"\n🔄 Processing {len(sample_ids)} samples...")
        
        reports = []
        
        # Process with progress bar
        for sample_id in tqdm(sample_ids, desc="Processing receipts"):
            try:
                report = self.process_single(sample_id, split, save_visualizations)
                reports.append(report)
            except Exception as e:
                print(f"\n⚠️  Error processing {sample_id}: {e}")
                continue
        
        return reports
    
    # ---------------------------
    # Generate summary statistics
    # ---------------------------
    def generate_summary(self, reports):
        """
        Generate summary statistics from all reports
        
        Args:
            reports: List of evaluation reports
        
        Returns:
            Summary statistics dict
        """
        if not reports:
            return {}
        
        df = pd.DataFrame(reports)
        
        summary = {
            'total_samples': int(len(reports)),
            'avg_text_similarity': float(df['text_similarity'].mean()),
            'avg_field_accuracy': float(df['field_accuracy'].mean()),
            'avg_confidence': float(df['avg_confidence'].mean()),
            'avg_review_percentage': float(df['review_percentage'].mean()),
            'total_words_reviewed': int(df['words_needing_review'].sum()),
            'low_conf_accuracy': float(df['low_conf_accuracy'].mean()),
            'high_conf_accuracy': float(df['high_conf_accuracy'].mean()),
        }
        
        return summary
    
    # ---------------------------
    # Save results
    # ---------------------------
    def save_results(self, reports, summary):
        """
        Save reports and summary to files
        
        Args:
            reports: List of reports
            summary: Summary statistics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed reports
        reports_file = self.output_dir / f"reports_{timestamp}.json"
        with open(reports_file, 'w') as f:
            json.dump(reports, f, indent=2)
        print(f"💾 Saved reports → {reports_file}")
        
        # Save summary
        summary_file = self.output_dir / f"summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"💾 Saved summary → {summary_file}")
        
        # Save as CSV for easy viewing
        df = pd.DataFrame(reports)
        csv_file = self.output_dir / f"results_{timestamp}.csv"
        df.to_csv(csv_file, index=False)
        print(f"💾 Saved CSV → {csv_file}")
    
    # ---------------------------
    # Print summary
    # ---------------------------
    def print_summary(self, summary):
        """Print formatted summary"""
        print("\n" + "=" * 60)
        print("📊 SUMMARY STATISTICS")
        print("=" * 60)
        print(f"Total Samples: {summary['total_samples']}")
        print()
        print(f"Avg Text Similarity: {summary['avg_text_similarity']:.1f}%")
        print(f"Avg Field Accuracy: {summary['avg_field_accuracy']:.1f}%")
        print(f"Avg OCR Confidence: {summary['avg_confidence']:.1f}%")
        print()
        print(f"Avg Words Needing Review: {summary['avg_review_percentage']:.1f}%")
        print(f"Total Words Reviewed: {summary['total_words_reviewed']:.0f}")
        print()
        print("Confidence Correlation:")
        print(f"  Low confidence → {summary['low_conf_accuracy']:.1f}% accurate")
        print(f"  High confidence → {summary['high_conf_accuracy']:.1f}% accurate")
        print()
        print("💡 KEY INSIGHT:")
        print(f"   By reviewing only {summary['avg_review_percentage']:.1f}% of words,")
        print(f"   you catch most errors (low conf = {summary['low_conf_accuracy']:.1f}% accurate)")
        print("=" * 60)


# =============================================================================
# Main execution
# =============================================================================
def main():
    """Main function"""
    print("=" * 60)
    print("🚀 CONFIDENCE-AWARE OCR PIPELINE")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = OCRPipeline("data/raw/SROIE2019")
    
    # Ask user how many to process
    print("\nHow many samples to process?")
    print("  1. Just 1 (quick test)")
    print("  2. First 10 (demo)")
    print("  3. First 50 (good sample)")
    print("  4. All 626 (full run - takes ~30 min)")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    samples_map = {
        '1': 1,
        '2': 10,
        '3': 50,
        '4': None  # All samples
    }
    
    num_samples = samples_map.get(choice, 1)
    
    # Ask about visualizations
    save_vis = input("\nSave visualizations? (y/n): ").strip().lower() == 'y'
    
    # Process samples
    reports = pipeline.process_batch(
        num_samples=num_samples,
        save_visualizations=save_vis
    )
    
    # Generate summary
    summary = pipeline.generate_summary(reports)
    
    # Print summary
    pipeline.print_summary(summary)
    
    # Save results
    pipeline.save_results(reports, summary)
    
    print("\n✅ Pipeline complete!")
    print(f"📁 Check results in: {pipeline.output_dir.absolute()}")


if __name__ == "__main__":
    main()