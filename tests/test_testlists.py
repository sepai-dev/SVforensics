import os
import unittest
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import tempfile
import shutil
from unittest.mock import patch, MagicMock

from svforensics import testlists


class TestListGenerationTests(unittest.TestCase):
    """Test cases for the test list generation functionality"""

    def setUp(self):
        """Set up test environment with mock data"""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock metadata
        self.metadata = pd.DataFrame({
            'file_path': [
                # Class 1 (male)
                'id001/video1/file1.wav', 'id001/video1/file2.wav', 
                'id001/video2/file1.wav', 'id001/video2/file2.wav',
                # Class 2 (male)
                'id002/video1/file1.wav', 'id002/video1/file2.wav',
                'id002/video2/file1.wav', 'id002/video2/file2.wav',
                # Class 3 (female)
                'id003/video1/file1.wav', 'id003/video1/file2.wav',
                'id003/video2/file1.wav', 'id003/video2/file2.wav',
                # Class 4 (female)
                'id004/video1/file1.wav', 'id004/video1/file2.wav',
                'id004/video2/file1.wav', 'id004/video2/file2.wav'
            ],
            'class_id': [
                'id001', 'id001', 'id001', 'id001',
                'id002', 'id002', 'id002', 'id002',
                'id003', 'id003', 'id003', 'id003',
                'id004', 'id004', 'id004', 'id004'
            ],
            'video_id': [
                'video1', 'video1', 'video2', 'video2',
                'video1', 'video1', 'video2', 'video2',
                'video1', 'video1', 'video2', 'video2',
                'video1', 'video1', 'video2', 'video2'
            ],
            'genre': [
                'm', 'm', 'm', 'm',
                'm', 'm', 'm', 'm',
                'f', 'f', 'f', 'f',
                'f', 'f', 'f', 'f'
            ]
        })
        
        # Create mock embeddings
        self.embeddings = {}
        for file_path in self.metadata['file_path']:
            self.embeddings[file_path] = torch.rand(192)  # Assume 192-dim embeddings
        
        # Create mock saved data
        self.saved_data = {
            'embeddings': self.embeddings,
            'metadata': self.metadata.to_dict('records')
        }
        
        # Path to the temporary embeddings file
        self.embeddings_path = os.path.join(self.temp_dir, 'embeddings.pth')
        
        # Save mock data
        torch.save(self.saved_data, self.embeddings_path)
        
        # Path for test list output
        self.output_prefix = os.path.join(self.temp_dir, 'test_list')

    def tearDown(self):
        """Clean up temporary files after tests"""
        shutil.rmtree(self.temp_dir)

    def test_load_processed_data(self):
        """Test loading processed data"""
        embeddings_dict, metadata_df = testlists.load_processed_data(self.embeddings_path)
        
        self.assertEqual(len(embeddings_dict), 16)
        self.assertEqual(len(metadata_df), 16)
        self.assertTrue('class_id' in metadata_df.columns)
        self.assertTrue('genre' in metadata_df.columns)

    def test_split_dataset(self):
        """Test splitting the dataset into reference and probe subsets"""
        # Perform split
        reference_df, probe_df = testlists.split_dataset(self.metadata, test_prop=0.5, random_state=42)
        
        # Check that each class appears in both reference and probe sets
        ref_classes = set(reference_df['class_id'].unique())
        probe_classes = set(probe_df['class_id'].unique())
        
        self.assertEqual(ref_classes, probe_classes)
        
        # Check that the sets have roughly equal size
        self.assertAlmostEqual(len(reference_df) / len(self.metadata), 0.5, delta=0.2)
        self.assertAlmostEqual(len(probe_df) / len(self.metadata), 0.5, delta=0.2)
        
        # Check that no file appears in both reference and probe sets
        ref_files = set(reference_df['file_path'])
        probe_files = set(probe_df['file_path'])
        
        self.assertEqual(len(ref_files.intersection(probe_files)), 0)

    def test_gender_filtering(self):
        """Test gender filtering in dataset split"""
        # Test male filtering
        reference_df_m, probe_df_m = testlists.split_dataset(
            self.metadata, test_prop=0.5, random_state=42, gender='m'
        )
        
        # Check that all entries are male
        self.assertTrue((reference_df_m['genre'] == 'm').all())
        self.assertTrue((probe_df_m['genre'] == 'm').all())
        
        # Test female filtering
        reference_df_f, probe_df_f = testlists.split_dataset(
            self.metadata, test_prop=0.5, random_state=42, gender='f'
        )
        
        # Check that all entries are female
        self.assertTrue((reference_df_f['genre'] == 'f').all())
        self.assertTrue((probe_df_f['genre'] == 'f').all())

    def test_generate_test_list_basic(self):
        """Test basic test list generation"""
        # Split dataset
        reference_df, probe_df = testlists.split_dataset(self.metadata, test_prop=0.5, random_state=42)
        
        # Generate test list with 1 positive and 1 negative trial per reference file
        # Note: different_videos is True by default now
        test_list = testlists.generate_test_list(
            reference_df, probe_df, n_pos=1, n_neg=1, random_state=42
        )
        
        # Check test list format
        self.assertTrue('label' in test_list.columns)
        self.assertTrue('reference' in test_list.columns)
        self.assertTrue('probe' in test_list.columns)
        
        # Check number of tests is correct
        expected_tests = len(reference_df) * 2  # 1 pos + 1 neg per reference
        self.assertEqual(len(test_list), expected_tests)
        
        # Check positive/negative distribution
        pos_count = (test_list['label'] == 1).sum()
        neg_count = (test_list['label'] == 0).sum()
        
        self.assertEqual(pos_count, len(reference_df))
        self.assertEqual(neg_count, len(reference_df))

    def test_no_duplicate_tests(self):
        """Test that there are no duplicate tests in the generated test list"""
        # Split dataset
        reference_df, probe_df = testlists.split_dataset(self.metadata, test_prop=0.5, random_state=42)
        
        # Generate test list with multiple trials
        test_list = testlists.generate_test_list(
            reference_df, probe_df, n_pos=2, n_neg=2, random_state=42
        )
        
        # Create unique test identifier by combining reference and probe
        test_list['test_id'] = test_list['reference'] + '_' + test_list['probe']
        
        # Check for duplicates
        duplicate_count = len(test_list) - len(test_list['test_id'].unique())
        self.assertEqual(duplicate_count, 0, "Found duplicate tests in the test list")

    def test_reference_probe_separation(self):
        """Test that reference files are only from reference set and probe files are only from probe set"""
        # Split dataset
        reference_df, probe_df = testlists.split_dataset(self.metadata, test_prop=0.5, random_state=42)
        
        ref_files = set(reference_df['file_path'])
        probe_files = set(probe_df['file_path'])
        
        # Generate test list
        test_list = testlists.generate_test_list(
            reference_df, probe_df, n_pos=2, n_neg=2, random_state=42
        )
        
        # Check that all reference files come from the reference set
        test_ref_files = set(test_list['reference'])
        self.assertTrue(test_ref_files.issubset(ref_files))
        
        # Check that all probe files come from the probe set
        test_probe_files = set(test_list['probe'])
        self.assertTrue(test_probe_files.issubset(probe_files))

    def test_same_videos_option(self):
        """Test that when same videos option is used, positive pairs can be from the same video"""
        # Split dataset
        reference_df, probe_df = testlists.split_dataset(self.metadata, test_prop=0.5, random_state=42)
        
        # Generate test list with different_videos=False to allow same videos
        test_list = testlists.generate_test_list(
            reference_df, probe_df, n_pos=1, n_neg=1, different_videos=False, random_state=42
        )
        
        # Extract positive pairs
        positive_pairs = test_list[test_list['label'] == 1]
        
        # Check that there is at least one positive pair from the same video
        same_video_pairs = 0
        for _, row in positive_pairs.iterrows():
            ref_file = row['reference']
            probe_file = row['probe']
            
            # Extract video IDs from file paths
            ref_video = ref_file.split('/')[1]
            probe_video = probe_file.split('/')[1]
            
            if ref_video == probe_video:
                same_video_pairs += 1
        
        # Not guaranteed to have same video pairs, but with our test data it should happen
        self.assertGreater(same_video_pairs, 0, "No positive pairs from the same video found when same_videos=True")

    def test_different_videos_default(self):
        """Test that the default behavior enforces different videos for positive pairs"""
        # Split dataset
        reference_df, probe_df = testlists.split_dataset(self.metadata, test_prop=0.5, random_state=42)
        
        # Generate test list with default settings (different_videos=True)
        test_list = testlists.generate_test_list(
            reference_df, probe_df, n_pos=1, n_neg=1, random_state=42
        )
        
        # Extract positive pairs
        positive_pairs = test_list[test_list['label'] == 1]
        
        # Check that for each positive pair, the video IDs are different
        for _, row in positive_pairs.iterrows():
            ref_file = row['reference']
            probe_file = row['probe']
            
            # Extract video IDs from file paths
            ref_video = ref_file.split('/')[1]
            probe_video = probe_file.split('/')[1]
            
            self.assertNotEqual(ref_video, probe_video, 
                               f"Found same video ID in positive pair: {ref_file} - {probe_file}")

    def test_class_separation_in_negative_pairs(self):
        """Test that negative pairs always have different class IDs"""
        # Split dataset
        reference_df, probe_df = testlists.split_dataset(self.metadata, test_prop=0.5, random_state=42)
        
        # Generate test list
        test_list = testlists.generate_test_list(
            reference_df, probe_df, n_pos=1, n_neg=1, random_state=42
        )
        
        # Extract negative pairs
        negative_pairs = test_list[test_list['label'] == 0]
        
        # Check that for each negative pair, the class IDs are different
        for _, row in negative_pairs.iterrows():
            ref_file = row['reference']
            probe_file = row['probe']
            
            # Extract class IDs from file paths
            ref_class = ref_file.split('/')[0]
            probe_class = probe_file.split('/')[0]
            
            self.assertNotEqual(ref_class, probe_class, 
                               f"Found same class ID in negative pair: {ref_file} - {probe_file}")

    def test_class_equality_in_positive_pairs(self):
        """Test that positive pairs always have the same class ID"""
        # Split dataset
        reference_df, probe_df = testlists.split_dataset(self.metadata, test_prop=0.5, random_state=42)
        
        # Generate test list
        test_list = testlists.generate_test_list(
            reference_df, probe_df, n_pos=1, n_neg=1, random_state=42
        )
        
        # Extract positive pairs
        positive_pairs = test_list[test_list['label'] == 1]
        
        # Check that for each positive pair, the class IDs are the same
        for _, row in positive_pairs.iterrows():
            ref_file = row['reference']
            probe_file = row['probe']
            
            # Extract class IDs from file paths
            ref_class = ref_file.split('/')[0]
            probe_class = probe_file.split('/')[0]
            
            self.assertEqual(ref_class, probe_class, 
                            f"Found different class IDs in positive pair: {ref_file} - {probe_file}")

    @patch('svforensics.testlists.load_processed_data')
    def test_create_test_lists_integration(self, mock_load_data):
        """Integration test for the create_test_lists function"""
        # Mock the load_processed_data function to return our test data
        mock_load_data.return_value = (self.embeddings, self.metadata)
        
        # Create test lists
        output_path = testlists.create_test_lists(
            gender="m",  # Providing a gender is now required
            embeddings_file=self.embeddings_path,
            output_prefix=self.output_prefix,
            n_pos=1,
            n_neg=1,
            different_videos=True,  # This is now the default
            test_prop=0.5,
            random_state=42
        )
        
        # Check that the output file exists
        self.assertTrue(os.path.exists(output_path))
        
        # Load and check the test list
        test_list = pd.read_csv(output_path, sep=' ', header=None, 
                                names=['label', 'reference', 'probe'])
        
        # Basic checks
        self.assertGreater(len(test_list), 0)
        self.assertTrue(set(test_list['label']).issubset({0, 1}))

    @patch('svforensics.testlists.load_processed_data')
    def test_create_test_lists_with_gender(self, mock_load_data):
        """Test test list creation with gender filtering"""
        # Mock the load_processed_data function to return our test data
        mock_load_data.return_value = (self.embeddings, self.metadata)
        
        # Create test lists with male gender filter
        output_path_m = testlists.create_test_lists(
            gender="m",  # Providing a gender is now required
            embeddings_file=self.embeddings_path,
            output_prefix=self.output_prefix,
            n_pos=1,
            n_neg=1,
            different_videos=True,  # This is now the default
            test_prop=0.5,
            random_state=42
        )
        
        # Create test lists with female gender filter
        output_path_f = testlists.create_test_lists(
            gender="f",  # Explicitly provide gender parameter
            embeddings_file=self.embeddings_path,
            output_prefix=self.output_prefix,
            n_pos=1,
            n_neg=1,
            different_videos=True,  # This is now the default
            test_prop=0.5,
            random_state=42
        )
        
        # Check that both files exist
        self.assertTrue(os.path.exists(output_path_m))
        self.assertTrue(os.path.exists(output_path_f))
        
        # Load and check the test lists
        test_list_m = pd.read_csv(output_path_m, sep=' ', header=None, 
                                 names=['label', 'reference', 'probe'])
        test_list_f = pd.read_csv(output_path_f, sep=' ', header=None, 
                                 names=['label', 'reference', 'probe'])
        
        # Check male test list
        for _, row in test_list_m.iterrows():
            ref_file = row['reference']
            ref_class = ref_file.split('/')[0]
            self.assertTrue(ref_class in ['id001', 'id002'], 
                           f"Found non-male class in male test list: {ref_class}")
        
        # Check female test list
        for _, row in test_list_f.iterrows():
            ref_file = row['reference']
            ref_class = ref_file.split('/')[0]
            self.assertTrue(ref_class in ['id003', 'id004'], 
                           f"Found non-female class in female test list: {ref_class}")


if __name__ == '__main__':
    unittest.main() 