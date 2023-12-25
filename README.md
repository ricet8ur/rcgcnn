
# This repository contains an approach to reproduce [article](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301) results. This is NOT an official implementation

For official implementation, see https://github.com/txie-93/cgcnn

1. Create conda environment and install requirements from RREADME.md
2. Download data from Materials Project.
   If we treat ids from given list (data/materials-data/mp-ids-46744.csv) as mp-ids, then mpr.materials.search(material_ids=ids) will return only 36420 of 46744 materials. 
   If we treat ids from given list as task-ids, it is possible to get corresponding mp-ids for them. It is reasonable to use this approach, since some of materials changed their mp-ids [https://discuss.matsci.org/t/change-in-materials-project-ids/1268](https://discuss.matsci.org/t/change-in-materials-project-ids/1268)
	
  
  With this, fetching function becomes: mpr.materials.search(task_ids=ids), according to mpr.get_material_id_from_task_id() inner structure.
	

  The returned mp-ids list contains 46296 entries, which is much better result.
	
	
  To properly check relationship between initial ids and returned mp-ids, a map: id => mp_id created with the help of mpr.materials.search(material_ids=mp-ids, fields=["material_id", "task_ids"]) -> list[(mp_id,list[task_id])]
	
	
  The map clearly shows that initial list of ids contains such pairs of task_ids, which correspond to the same mp-id, and the total number of duplicates is 830 entries (~1.8% of dataset).
3. Train without hyperparameter optimization
4. Train with hyperparameter optimization
   
#### Performance

During the reproduction of the results of the paper, performance analysis was made. It shows that in initial implementation the zero epoch training is taking significant amount of time (up to half of total train time on the user machine). With the help of flamegraph slow parts of the code were found and improved:


- Now all data (structures and properties) is stored in binary msgpack files for ease of archiving and speed.
- Replaced structures read function: unpacking from dict, not from string
- Improved CifData.get_item() with the help of numba: features calculation functions was replaced with numba-compiled versions (expand() and, partially, get_all_neighbours())
- The crucial optimization part is caching of feature tensors in the CifData.get_item(). Therefore, main() was modified so that it is possible to call it from other python script with the same required args while the cache could be saved between the main() calls. (Note: different dataset and hyperparams can make the cache to be recreated)
- Multiprocessing during dataset loading with -j option does not work in this implementation (yet?) 
  
#### Reproducibility of training

Original implementation does not provide deterministic learning option.


This option is implemented in this repo and now possible with torch_generator argument (set it to any value to get deterministic learning). To make it possible, some changes were made to main() and data.py (DataLoader part).
