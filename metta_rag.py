"""
MeTTa RAG for Bimanual Dataset Knowledge Graph
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
from hyperon import MeTTa, E, S, ValueAtom


class BimanualDatasetRAG:
    """RAG system using MeTTa for bimanual dataset knowledge graph"""
    
    def __init__(self):
        self.metta = MeTTa()
        print("✓ MeTTa knowledge graph initialized")
    
    def add_dataset(self, task: str, ipfs_hash: str, metadata: Dict):
        """
        Add a dataset to the knowledge graph
        
        Args:
            task: Task name (e.g., "opening_bottle")
            ipfs_hash: IPFS hash of the dataset
            metadata: Dictionary with quality_score, duration, etc.
        """
        # Normalize task name
        task = task.lower().strip().replace(' ', '_')
        
        # Add to graph: (dataset task_name ipfs_hash)
        self.metta.space().add_atom(E(S("dataset"), S(task), S(ipfs_hash)))
        
        # Add metadata
        for key, value in metadata.items():
            if isinstance(value, (int, float)):
                self.metta.space().add_atom(
                    E(S("metadata"), S(ipfs_hash), S(key), ValueAtom(value))
                )
            else:
                self.metta.space().add_atom(
                    E(S("metadata"), S(ipfs_hash), S(key), ValueAtom(str(value)))
                )
        
        print(f"✓ Added dataset to knowledge graph: {task} → {ipfs_hash}")
    
    def query_datasets_by_task(self, task: str) -> List[str]:
        """
        Query datasets for a given task (handles plurals and variations)
        
        Args:
            task: Task name (e.g., "opening_bottle" or "opening_bottles")
            
        Returns:
            List of IPFS hashes
        """
        # Normalize the task name
        task = task.lower().strip()
        task = task.replace(' ', '_')
        
        # Create variations to try (singular and plural)
        task_variations = [task]
        
        # Add singular form if ends with 's'
        if task.endswith('s') and not task.endswith('ss'):
            singular = task[:-1]
            if singular not in task_variations:
                task_variations.append(singular)
        
        # Add plural form if doesn't end with 's'
        if not task.endswith('s'):
            plural = task + 's'
            if plural not in task_variations:
                task_variations.append(plural)
        
        # Try each variation
        all_results = []
        for task_variant in task_variations:
            query_str = f'!(match &self (dataset {task_variant} $hash) $hash)'
            results = self.metta.run(query_str)
            
            if results and len(results) > 0 and results[0]:
                for result in results[0]:
                    try:
                        # Convert to string
                        ipfs_hash = str(result)
                        if ipfs_hash and ipfs_hash not in all_results:
                            all_results.append(ipfs_hash)
                    except Exception as e:
                        # Skip problematic results
                        continue
        
        return all_results
    
    def get_dataset_metadata(self, ipfs_hash: str) -> Dict:
        """Get metadata for a specific dataset"""
        metadata = {}
        
        query_str = f'!(match &self (metadata {ipfs_hash} $key $value) (list $key $value))'
        results = self.metta.run(query_str)
        
        if results and len(results) > 0 and results[0]:
            for result in results[0]:
                try:
                    # Result should be a list/tuple with [key, value]
                    if hasattr(result, '__len__') and len(result) >= 2:
                        key = str(result[0])
                        value = result[1]
                        
                        # Extract the actual value
                        if hasattr(value, 'value'):
                            metadata[key] = value.value()
                        else:
                            metadata[key] = str(value)
                    else:
                        # Single ExpressionAtom - skip
                        continue
                except Exception as e:
                    # Skip problematic results
                    continue
        
        return metadata
    
    def get_all_tasks(self) -> List[str]:
        """Get all unique tasks in the knowledge graph"""
        query_str = '!(match &self (dataset $task $hash) $task)'
        results = self.metta.run(query_str)
        
        if results and results[0]:
            tasks = list(set([str(task) for task in results[0]]))
            return sorted(tasks)
        return []
    
    def find_similar_tasks(self, task: str, max_results: int = 5) -> List[str]:
        """
        Find similar tasks using simple string matching
        
        Args:
            task: Task to find similar ones for
            max_results: Maximum number of results
            
        Returns:
            List of similar task names
        """
        task = task.lower().strip().replace(' ', '_')
        all_tasks = self.get_all_tasks()
        
        # Score each task by similarity
        scored_tasks = []
        for existing_task in all_tasks:
            existing_lower = existing_task.lower()
            
            # Exact match (shouldn't happen, but just in case)
            if existing_lower == task:
                continue
            
            # Calculate similarity score
            score = 0
            
            # Check if task words are in existing task
            task_words = task.split('_')
            for word in task_words:
                if word in existing_lower:
                    score += 2
            
            # Check if existing task words are in query
            existing_words = existing_lower.split('_')
            for word in existing_words:
                if word in task:
                    score += 1
            
            # Partial substring match
            if task in existing_lower or existing_lower in task:
                score += 3
            
            if score > 0:
                scored_tasks.append((score, existing_task))
        
        # Sort by score and return top results
        scored_tasks.sort(reverse=True, key=lambda x: x[0])
        return [task for score, task in scored_tasks[:max_results]]
    
    def get_category_for_task(self, task: str) -> Optional[str]:
        """Get category for a given task from config"""
        from config import Config
        
        task = task.lower().strip().replace(' ', '_')
        
        for category, tasks in Config.TASK_TAXONOMY.items():
            if task in tasks:
                return category
        
        return None
    
    def get_statistics(self) -> Dict:
        """Get knowledge graph statistics"""
        all_tasks = self.get_all_tasks()
        
        stats = {
            'total_tasks': len(all_tasks),
            'total_datasets': 0,
            'tasks_by_category': {}
        }
        
        # Count total datasets
        for task in all_tasks:
            datasets = self.query_datasets_by_task(task)
            stats['total_datasets'] += len(datasets)
            
            # Count by category
            category = self.get_category_for_task(task)
            if category:
                stats['tasks_by_category'][category] = stats['tasks_by_category'].get(category, 0) + 1
        
        return stats
    
    def export_knowledge_graph(self, output_path: Path):
        """Export knowledge graph to JSON"""
        all_tasks = self.get_all_tasks()
        
        export_data = {
            'tasks': {},
            'metadata': {
                'total_tasks': len(all_tasks),
                'export_timestamp': None
            }
        }
        
        for task in all_tasks:
            try:
                datasets = self.query_datasets_by_task(task)
                export_data['tasks'][task] = {
                    'datasets': []
                }
                
                for ipfs_hash in datasets:
                    try:
                        metadata = self.get_dataset_metadata(ipfs_hash)
                        export_data['tasks'][task]['datasets'].append({
                            'ipfs_hash': ipfs_hash,
                            'metadata': metadata
                        })
                    except Exception as e:
                        # Skip datasets with errors
                        print(f"⚠️  Skipping dataset {ipfs_hash}: {e}")
                        continue
            except Exception as e:
                # Skip tasks with errors
                print(f"⚠️  Skipping task {task}: {e}")
                continue
        
        # Add timestamp
        from datetime import datetime
        export_data['metadata']['export_timestamp'] = datetime.now().isoformat()
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"✓ Knowledge graph exported to: {output_path}")
    
    def import_knowledge_graph(self, input_path: Path):
        """Import knowledge graph from JSON"""
        if not input_path.exists():
            print(f"⚠️  No existing knowledge graph found at: {input_path}")
            return
        
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON formats
            tasks_data = data.get('tasks', {})
            
            # Check if tasks_data is a dict or list
            if isinstance(tasks_data, list):
                # Old format - skip
                print(f"⚠️  Old JSON format detected, skipping import")
                return
            
            total_loaded = 0
            
            for task, task_info in tasks_data.items():
                # Handle case where task_info might be a list
                if isinstance(task_info, list):
                    datasets = task_info
                elif isinstance(task_info, dict):
                    datasets = task_info.get('datasets', [])
                else:
                    continue
                
                for dataset in datasets:
                    try:
                        # Handle different dataset formats
                        if isinstance(dataset, str):
                            ipfs_hash = dataset
                            metadata = {}
                        elif isinstance(dataset, dict):
                            ipfs_hash = dataset.get('ipfs_hash')
                            metadata = dataset.get('metadata', {})
                        else:
                            continue
                        
                        if ipfs_hash:
                            # Add dataset back to graph
                            self.metta.space().add_atom(E(S("dataset"), S(task), S(ipfs_hash)))
                            
                            # Add metadata back
                            for key, value in metadata.items():
                                if isinstance(value, (int, float)):
                                    self.metta.space().add_atom(
                                        E(S("metadata"), S(ipfs_hash), S(key), ValueAtom(value))
                                    )
                                else:
                                    self.metta.space().add_atom(
                                        E(S("metadata"), S(ipfs_hash), S(key), ValueAtom(str(value)))
                                    )
                            
                            total_loaded += 1
                    except Exception as e:
                        # Skip problematic datasets
                        print(f"⚠️  Skipping dataset: {e}")
                        continue
            
            print(f"✓ Loaded {total_loaded} dataset(s) from knowledge graph")
            
        except Exception as e:
            print(f"⚠️  Error loading knowledge graph: {e}")