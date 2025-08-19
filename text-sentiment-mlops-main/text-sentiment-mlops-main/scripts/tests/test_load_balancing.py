import requests
import time
import logging
from typing import Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LoadBalancingTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.instances_seen = set()
        self.response_times = []
        self.instance_stats = {}

    def test_health_checks(self) -> bool:
        """Test health check endpoint across all instances"""
        logger.info("Testing health checks...")
        try:
            response = requests.get(f"{self.base_url}/health")
            response.raise_for_status()
            data = response.json()
            
            # Verify instance information
            assert "instance_info" in data, "Instance info missing in health check response"
            assert "id" in data["instance_info"], "Instance ID missing in health check response"
            assert "hostname" in data["instance_info"], "Hostname missing in health check response"
            assert "ip_address" in data["instance_info"], "IP address missing in health check response"
            
            logger.info(f"Health check successful for instance {data['instance_info']['id']}")
            return True
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return False

    def test_prediction(self, text: str = "I love this product!") -> Dict:
        """Test prediction endpoint and return response with instance info"""
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json={"text": text}
            )
            response.raise_for_status()
            data = response.json()
            
            # Verify instance information
            assert "instance_info" in data, "Instance info missing in prediction response"
            assert "id" in data["instance_info"], "Instance ID missing in prediction response"
            assert "hostname" in data["instance_info"], "Hostname missing in prediction response"
            assert "ip_address" in data["instance_info"], "IP address missing in prediction response"
            
            return data
        except Exception as e:
            logger.error(f"Prediction request failed: {str(e)}")
            return None

    def test_load_distribution(self, num_requests: int = 10) -> Dict:
        """Test load distribution across instances"""
        logger.info(f"Testing load distribution with {num_requests} requests...")
        
        for i in range(num_requests):
            response = self.test_prediction()
            
            if response:
                instance_id = response["instance_info"]["id"]
                self.instances_seen.add(instance_id)
                
                # Track response times
                response_time = response["response_time_ms"]
                self.response_times.append(response_time)
                
                # Track instance statistics
                if instance_id not in self.instance_stats:
                    self.instance_stats[instance_id] = {
                        "requests": 0,
                        "total_time": 0,
                        "avg_time": 0
                    }
                self.instance_stats[instance_id]["requests"] += 1
                self.instance_stats[instance_id]["total_time"] += response_time
                self.instance_stats[instance_id]["avg_time"] = self.instance_stats[instance_id]["total_time"] / self.instance_stats[instance_id]["requests"]
            
            # Small delay between requests
            time.sleep(0.5)
        
        return {
            "instances_seen": list(self.instances_seen),
            "instance_stats": self.instance_stats,
            "avg_response_time": sum(self.response_times) / len(self.response_times) if self.response_times else 0
        }

    def print_test_results(self, results: Dict):
        """Print formatted test results"""
        logger.info("\n=== Load Balancing Test Results ===")
        logger.info(f"Number of instances seen: {len(results['instances_seen'])}")
        logger.info(f"Instances: {results['instances_seen']}")
        logger.info(f"Average response time: {results['avg_response_time']:.2f}ms")
        
        logger.info("\nInstance Statistics:")
        for instance_id, stats in results['instance_stats'].items():
            logger.info(f"\nInstance {instance_id}:")
            logger.info(f"  Requests handled: {stats['requests']}")
            logger.info(f"  Average response time: {stats['avg_time']:.2f}ms")

def main():
    # Initialize tester
    tester = LoadBalancingTester()
    
    # Test health checks
    logger.info("\n=== Testing Health Checks ===")
    health_check_success = tester.test_health_checks()
    if not health_check_success:
        logger.error("Health check tests failed")
        return
    
    # Test load distribution
    logger.info("\n=== Testing Load Distribution ===")
    results = tester.test_load_distribution(num_requests=20)
    
    # Print results
    tester.print_test_results(results)
    
    # Verify load balancing
    if len(results['instances_seen']) > 1:
        logger.info("\n✅ Load balancing is working - requests distributed across multiple instances")
    else:
        logger.warning("\n⚠️ Load balancing might not be working - all requests handled by single instance")

if __name__ == "__main__":
    main()
