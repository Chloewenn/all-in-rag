--> 正在初始化编码器和Milvus客户端...

--> 正在创建 Collection 'multimodal_demo'
Schema 结构:
{'auto_id': True, 'description': '多模态图文检索', 'fields': [{'name': 'id', 'description': '', 'type': <DataType.INT64: 5>, 'is_primary': True, 'auto_id': True}, {'name': 'vector', 'description': '', 'type': <DataType.FLOAT_VECTOR: 101>, 'params': {'dim': 768}}, {'name': 'image_path', 'description': '', 'type': <DataType.VARCHAR: 21>, 'params': {'max_length': 512}}], 'enable_dynamic_field': False}
成功创建 Collection: 'multimodal_demo'
Collection 结构:
{'collection_name': 'multimodal_demo', 'auto_id': True, 'num_shards': 1, 'description': '多模态图文检索', 'fields': [{'field_id': 100, 'name': 'id', 'description': '', 'type': <DataType.INT64: 5>, 'params': {}, 'auto_id': True, 'is_primary': True}, {'field_id': 101, 'name': 'vector', 'description': '', 'type': <DataType.FLOAT_VECTOR: 101>, 'params': {'dim': 768}}, {'field_id': 102, 'name': 'image_path', 'description': '', 'type': <DataType.VARCHAR: 21>, 'params': {'max_length': 512}}], 'functions': [], 'aliases': [], 'collection_id': 469773526962537045, 'consistency_level': 2, 'properties': {}, 'num_partitions': 1, 'enable_dynamic_field': False, 'created_timestamp': 469773557819244550, 'update_timestamp': 469773557819244550}

--> 正在向 'multimodal_demo' 插入数据
生成图像嵌入: 100%|███████████████████████████████████████████████| 7/7 [00:01<00:00,  4.37it/s]
成功插入 7 条数据。

--> 正在为 'multimodal_demo' 创建索引
成功为向量字段创建 HNSW 索引。
索引详情:
{'M': '16', 'efConstruction': '256', 'metric_type': 'COSINE', 'index_type': 'HNSW', 'field_name': 'vector', 'index_name': 'vector', 'total_rows': 0, 'indexed_rows': 0, 'pending_index_rows': 0, 'state': 'Finished'}
已加载 Collection 到内存中。

--> 正在 'multimodal_demo' 中执行检索
检索结果:
  Top 1: ID=469773526962537061, 距离=0.9466, 路径='../../data/C3/dragon/query.png'
  Top 2: ID=469773526962537066, 距离=0.7443, 路径='../../data/C3/dragon/dragon02.png'
  Top 3: ID=469773526962537063, 距离=0.6851, 路径='../../data/C3/dragon/dragon06.png'
  Top 4: ID=469773526962537064, 距离=0.6049, 路径='../../data/C3/dragon/dragon03.png'
  Top 5: ID=469773526962537060, 距离=0.5360, 路径='../../data/C3/dragon/dragon05.png'

--> 正在可视化结果并清理资源
结果图像已保存到: ../../data/C3/search_result.png
已从内存中释放 Collection: 'multimodal_demo'
已删除 Collection: 'multimodal_demo'
