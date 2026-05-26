from elasticsearch import Elasticsearch
import numpy as np

count = 0

class ElasticSearchManager:
    def __init__(self, index_name: str, state_dim: int):
        """初始化ES管理器
        
        Args:
            index_name: ES索引名称
            state_dim: 状态向量维度
        """
        clean_index_name = "".join(c.lower() for c in index_name if c.isalnum() or c == '_' or c == '-')
        
        self.es = Elasticsearch(
            # ["node01:9200"],
            ["127.0.0.1:10352"],
            sniff_on_start=False,          
            sniff_on_connection_fail=False, 
            timeout=120, 
            max_retries=10, 
            retry_on_timeout=True           
        )
        self.index_name = clean_index_name
        
        mapping = {
            "mappings": {
                "properties": {
                    "state": {
                        "type": "dense_vector",
                        "dims": state_dim
                    },
                    "metadata": {
                        "properties": {
                            "path_ind": {"type": "integer"},
                            "timestep": {"type": "integer"},
                            "discounted_return": {"type": "float"},
                            "observations": {
                                "type": "dense_vector",
                                "dims": state_dim
                            }
                        }
                    }
                }
            }
        }
        
        if not self.es.indices.exists(index=index_name):
            self.es.indices.create(index=index_name, body=mapping)
    

    def find_similar_states(self, query_state: np.ndarray, k: int = 10, min_score: float = 0):
        """查找相似状态"""
        global count
        try:
            if not isinstance(query_state, np.ndarray):
                raise ValueError("query_state必须是numpy数组")
            
            query = {
                "size": k,
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'state') + 1.0",
                            "params": {
                                "query_vector": query_state.tolist()
                            }
                        }
                    }
                },
                "_source": ["metadata"]
            }
            
            response = self.es.search(
                index=self.index_name,
                body=query
            )
            
            if 'hits' not in response or 'hits' not in response['hits']:
                print("无效的响应格式")
                return []
            
            return [
                {
                    'similarity': hit['_score'],
                    'metadata': hit['_source']['metadata']
                }
                for hit in response['hits']['hits']
                if hit['_score'] >= min_score and hit['_source']['metadata']["discounted_return"] >= 0
            ]
            
        except Exception as e:
            count += 1
            print(f"count: {count}")
            return []
        

    def find_similar_states_nok(self, query_state: np.ndarray, min_score: float = 0, batch_size: int = 1000):
      
        global count
        scroll_id = None
        try:
            if not isinstance(query_state, np.ndarray):
                raise ValueError("query_state必须是numpy数组")

            query = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'state') + 1.0",
                            "params": {
                                "query_vector": query_state.tolist()
                            }
                        }
                    }
                },
                "min_score": min_score,
                "_source": ["metadata"]
            }

            all_hits = []
            response = self.es.search(
                index=self.index_name,
                body=query,
                scroll="1m",    
                size=batch_size
            )

            scroll_id = response.get("_scroll_id")
            hits = response["hits"]["hits"]
            all_hits.extend(hits)

            while hits:
                response = self.es.scroll(scroll_id=scroll_id, scroll="1m")
                scroll_id = response.get("_scroll_id")
                hits = response["hits"]["hits"]
                if not hits:
                    break
                all_hits.extend(hits)

            return [
                {
                    'similarity': hit['_score'],
                    'metadata': hit['_source']['metadata']
                }
                for hit in all_hits
                if hit['_source']['metadata']["discounted_return"] >= 0
            ]

        except Exception as e:
            count += 1
            print(f"count: {count}, error: {e}")
            return []

        finally:
            if scroll_id:
                try:
                    self.es.clear_scroll(scroll_id=scroll_id)
                except Exception as clear_e:
                    print(f"清理 scroll_id 出错: {clear_e}")

  


class ElasticSearchManager1:
    def __init__(self, index_name: str, state_dim: int):
        """初始化ES管理器
        
        Args:
            index_name: ES索引名称
            state_dim: 状态向量维度
        """
        clean_index_name = "".join(c.lower() for c in index_name if c.isalnum() or c == '_' or c == '-')
        
        self.es = Elasticsearch(
            ["127.0.0.1:10352"],
            sniff_on_start=False,         
            sniff_on_connection_fail=False, 
            timeout=120, 
            max_retries=10, 
            retry_on_timeout=True             
        )
        self.index_name = clean_index_name
        
        # 优化索引映射
        # 定义ES索引的映射
        mapping = {
            "mappings": {
                "properties": {
                
                    "state": {
                        "type": "dense_vector",
                        "dims": state_dim 
                    },
                    "state_x": { "type": "float" },
                    "state_y": { "type": "float" },
                    "metadata": {
                        "properties": {
                            "path_ind": {"type": "integer"},
                            "timestep": {"type": "integer"},
                            "discounted_return": {"type": "float"},
                            "observations": {
                                "type": "dense_vector",
                                "dims": state_dim
                            }
                        }
                    }
                }
            }
        }
        
        if not self.es.indices.exists(index=index_name):
            self.es.indices.create(index=index_name, body=mapping)
    

    def find_similar_states_antmaze_first2(self, query_state: np.ndarray, k: int = 10, min_score: float = 0, candidate_size: int = 200):
        global count
        try:
            if not isinstance(query_state, np.ndarray):
                raise ValueError("query_state必须是numpy数组")
            
            query_first2 = query_state[:2].tolist()
            qx, qy = query_first2

            query = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            # 手工算 L2: 1 / (1 + (dx^2 + dy^2)^0.5)
                            "source": """
                                double dx = doc['state_x'].value  - params.qx;
                                double dy = doc['state_y'].value  - params.qy;
                                double dist = Math.sqrt(dx*dx + dy*dy);
                                return 1.0 / (1.0 + dist);
                            """,
                            "params": {"qx": qx, "qy": qy}
                        }
                    }
                },
                "min_score": 0.7,   
                "_source": ["state", "metadata"]
            }


            response = self.es.search(index=self.index_name, body=query)

            if 'hits' not in response or 'hits' not in response['hits']:
                print("无效的响应格式")
                return []

            candidates = [hit for hit in response['hits']['hits'] if hit['_score'] >= 0.7]
        

            scored_candidates = []
            for hit in candidates:
                cand_state = np.array(hit['_source']['state'])
                cos_sim = np.dot(query_state, cand_state) / (
                    np.linalg.norm(query_state) * np.linalg.norm(cand_state) + 1e-8
                )
                scored_candidates.append({
                    'similarity': float(cos_sim),
                    'metadata': hit['_source']['metadata']
                })

            scored_candidates = [c for c in scored_candidates if c['similarity'] >= min_score]
            scored_candidates.sort(key=lambda x: x['similarity'], reverse=True)

            return scored_candidates[:k]
        
        except Exception as e:
            count += 1
            print(f"count: {count}, error: {e}")
            return []
        


    def find_similar_states_nok_antmaze_first2(self, query_state: np.ndarray, min_score: float = 0, batch_size: int = 1000, top_m: int = 2000):
        global count
        scroll_id = None
        try:
            if not isinstance(query_state, np.ndarray):
                raise ValueError("query_state必须是numpy数组")
            
            qx, qy = query_state[:2].tolist()
            query = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": """
                                if (!doc['state_x'].empty && !doc['state_y'].empty) {
                                    double dx = doc['state_x'].value - params.qx;
                                    double dy = doc['state_y'].value - params.qy;
                                    double dist = Math.sqrt(dx*dx + dy*dy);
                                    return 1.0 / (1.0 + dist);
                                } else {
                                    return 0.0;
                                }
                            """,
                            "params": {"qx": qx, "qy": qy}
                        }
                    }
                },
                "min_score": 0.7,   
                "_source": ["state", "metadata"]
            }

            all_hits = []
            response = self.es.search(
                index=self.index_name,
                body=query,
                scroll="1m",
                size=batch_size
            )
            scroll_id = response["_scroll_id"]
            hits = response["hits"]["hits"]
            all_hits.extend(hits)

            while hits:
                response = self.es.scroll(scroll_id=scroll_id, scroll="1m")
                scroll_id = response["_scroll_id"]
                hits = response["hits"]["hits"]
                if not hits:
                    break
                all_hits.extend(hits)


            scored_candidates = []
            for hit in all_hits:
                cand_state = np.array(hit['_source']['state'], dtype=np.float32)
                cos_sim = np.dot(query_state, cand_state) / (np.linalg.norm(query_state) * np.linalg.norm(cand_state) + 1e-8)
                scored_candidates.append({
                    'similarity': float(cos_sim),
                    'metadata': hit['_source']['metadata']
                })

            scored_candidates = [c for c in scored_candidates if c['similarity'] >= min_score]
            scored_candidates.sort(key=lambda x: x['similarity'], reverse=True)

            return scored_candidates

        except Exception as e:
            count += 1
            print(f"count: {count}, error: {e}")
            return []

        finally:
            if scroll_id is not None:
                try:
                    self.es.clear_scroll(scroll_id=scroll_id)
                except:
                    pass

