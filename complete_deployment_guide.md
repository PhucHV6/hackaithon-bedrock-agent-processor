# AWS Bedrock Flows File Processor - Complete Setup Guide

This comprehensive guide will help you build, deploy, and use AWS Bedrock Flows for processing multiple file types (TXT, PDF, JSON, CSV, MD) through a Streamlit web interface.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   AWS Bedrock    │    │   Foundation    │
│   Web App       │───▶│   Flows          │───▶│   Models        │
│                 │    │                  │    │   (Claude,etc)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐    ┌──────────────────┐
│   File Upload   │    │   Flow Execution │
│   Processing    │    │   & Results      │
└─────────────────┘    └──────────────────┘
```

## 🚀 Quick Start (5 Minutes)

### Prerequisites
- AWS Account with Bedrock access
- Docker installed
- AWS CLI configured (optional)

### 1. Project Setup
```bash
# Create project directory
mkdir bedrock-flows-processor
cd bedrock-flows-processor

# Create the file structure
mkdir -p .streamlit
```

### 2. Create All Required Files

Copy these files from the artifacts:
- `enhanced_streamlit_app.py` (main application)
- `flow_builder.py` (flow creation script)
- `Dockerfile`
- `docker-compose.yml`
- `requirements.txt`
- `.streamlit/config.toml`

### 3. Quick Deploy
```bash
# Build and run
docker-compose up --build -d

# Access the app
open http://localhost:8501
```

## 📋 Step-by-Step Setup

### Step 1: AWS Bedrock Setup

#### 1.1 Enable Bedrock Models
```bash
# Check available models in your region
aws bedrock list-foundation-models --region us-east-1

# Request access to Claude models if needed
# Go to AWS Console > Bedrock > Model Access
```

#### 1.2 Create IAM Role for Flows
```bash
# Create the role using the flow_builder.py script
python flow_builder.py
```

Or manually in AWS Console:
```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:InvokeModel",
                "bedrock:InvokeModelWithResponseStream",
                "bedrock:ListFoundationModels"
            ],
            "Resource": "*"
        }
    ]
}
```

#### 1.3 Create Your First Flow

**Option A: Use the Flow Builder Script**
```python
from flow_builder import BedrockFlowBuilder

builder = BedrockFlowBuilder(region_name='us-east-1')

# Create IAM role
role_arn = create_iam_role_for_flows()

# Create file processing flow
file_flow = builder.create_file_processing_flow(
    flow_name="FileProcessingFlow",
    execution_role_arn=role_arn
)

print(f"Flow ID: {file_flow['flow']['id']}")
print(f"Alias ID: {file_flow['alias']['id']}")
```

**Option B: AWS Console**
1. Go to AWS Bedrock > Flows
2. Click "Create Flow"
3. Design your flow with these nodes:
   - Input Node (FlowInput)
   - Prompt Node (for file type detection)
   - Prompt Node (for content processing)
   - Output Node (ProcessedOutput)

### Step 2: Flow Design Patterns

#### 2.1 Basic File Processing Flow Structure

```
[Input] → [File Type Detector] → [Content Processor] → [Output]
```

**Input Node Configuration:**
- Name: `FlowInput`
- Output: `document` (Object)

**File Type Detector Node:**
- Type: Prompt
- Model: Claude 3 Sonnet
- Template: Analyze file type and structure
- Inputs: file_name, file_content
- Output: file_analysis

**Content Processor Node:**
- Type: Prompt  
- Model: Claude 3 Sonnet
- Template: Process content based on file type
- Inputs: file_analysis, file_content, user_query
- Output: processed_result

#### 2.2 Advanced Flow with Conditional Processing

```
[Input] → [File Type Router] → [JSON Processor]
                           ├→ [CSV Analyzer]
                           ├→ [PDF Extractor]
                           └→ [Text Summarizer] → [Output]
```

### Step 3: Application Deployment

#### 3.1 Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

#### 3.2 Docker Deployment
```bash
# Build image
docker build -t bedrock-flows-app .

# Run container
docker run -p 8501:8501 \
  -e AWS_ACCESS_KEY_ID=your_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret \
  bedrock-flows-app
```

#### 3.3 Production Deployment

**Using Docker Compose:**
```yaml
version: '3.8'
services:
  bedrock-app:
    build: .
    ports:
      - "8501:8501"
    environment:
      - AWS_DEFAULT_REGION=us-east-1
    volumes:
      - ~/.aws:/root/.aws:ro
    restart: unless-stopped
```

**Using AWS ECS:**
```bash
# Create ECS task definition
aws ecs register-task-definition \
  --family bedrock-flows-app \
  --task-role-arn arn:aws:iam::ACCOUNT:role/ecsTaskRole \
  --container-definitions file://task-definition.json
```

## 🎯 Usage Guide

### File Upload and Processing

#### 1. Supported File Types

| File Type | Extension | Processing Features |
|-----------|-----------|-------------------|
| **Text** | .txt | Word count, line analysis, content summarization |
| **PDF** | .pdf | Text extraction, page count, content analysis |
| **JSON** | .json | Structure analysis, data validation, key extraction |
| **CSV** | .csv | Data analysis, column insights, statistical summary |
| **Markdown** | .md | Header extraction, structure analysis, content parsing |

#### 2. File Processing Workflow

```
Upload File → File Analysis → Content Extraction → AI Processing → Results
```

**Step 1: Upload**
- Drag and drop or browse for files
- File validation and size checking
- File type detection

**Step 2: Analysis**
- Extract metadata (size, structure, etc.)
- Content preview generation
- Processing strategy determination

**Step 3: Processing**
- Content extraction based on file type
- Structure analysis for structured data
- Text normalization and cleaning

**Step 4: AI Processing**
- Send to Bedrock Flow
- Apply user queries and instructions
- Generate insights and summaries

### Example Use Cases

#### 1. Document Analysis
```python
# Upload: research_paper.pdf
# Query: "Summarize the key findings and methodology"
# Result: Structured summary with main points
```

#### 2. Data Insights
```python
# Upload: sales_data.csv
# Query: "What are the top trends and anomalies in this data?"
# Result: Statistical analysis and trend identification
```

#### 3. Code Review
```python
# Upload: config.json
# Query: "Validate this configuration and suggest improvements"
# Result: Configuration analysis and recommendations
```

#### 4. Content Processing
```python
# Upload: documentation.md
# Query: "Create a table of contents and identify missing sections"
# Result: Structured outline and gap analysis
```

## 🔧 Advanced Configuration

### Flow Customization

#### 1. Custom Prompt Templates

**File Type Detection Template:**
```text
Analyze the following file information:

File Name: {{file_name}}
Content Preview: {{file_content}}

Determine:
1. File type and format
2. Content structure and organization
3. Key data elements or sections
4. Processing recommendations

Response format: JSON with structure analysis
```

**Content Processing Template:**
```text
Process this {{file_type}} file based on the user's request:

File Analysis: {{file_analysis}}
Content: {{file_content}}
User Query: {{user_query}}

Instructions:
- For CSV: Provide data insights, trends, and statistics
- For JSON: Validate structure and extract key information
- For PDF/TXT: Summarize content and answer specific questions
- For MD: Analyze document structure and content organization

Provide comprehensive analysis with actionable insights.
```

#### 2. Model Configuration

**Recommended Models by Use Case:**

| Use Case | Model | Temperature | Max Tokens |
|----------|-------|-------------|------------|
| **Data Analysis** | Claude 3 Sonnet | 0.1 | 2000 |
| **Content Summary** | Claude 3 Sonnet | 0.3 | 1500 |
| **Creative Writing** | Claude 3 Opus | 0.7 | 3000 |
| **Code Analysis** | Claude 3 Sonnet | 0.2 | 2500 |

#### 3. Flow Optimization

**Performance Tips:**
- Use appropriate model sizes for your use case
- Implement content chunking for large files
- Add caching for repeated operations
- Use parallel processing for multiple files

### Security and Compliance

#### 1. Data Privacy
```python
# Implement data encryption
inputs = {
    'content': encrypt_content(file_content),
    'query': user_query,
    'encryption_key': generate_key()
}
```

#### 2. Access Control
```python
# Role-based access
def check_permissions(user_role, operation):
    permissions = {
        'admin': ['create', 'read', 'update', 'delete'],
        'user': ['read', 'upload'],
        'viewer': ['read']
    }
    return operation in permissions.get(user_role, [])
```

#### 3. Audit Logging
```python
# Log all operations
def log_operation(user_id, operation, file_info):
    log_entry = {
        'timestamp': datetime.now(),
        'user_id': user_id,
        'operation': operation,
        'file_info': file_info,
        'status': 'success'
    }
    audit_logger.info(json.dumps(log_entry))
```

## 🔍 Monitoring and Troubleshooting

### Application Monitoring

#### 1. Health Checks
```python
def health_check():
    checks = {
        'bedrock_connection': test_bedrock_connection(),
        'flow_availability': check_flow_status(),
        'model_access': verify_model_access()
    }
    return all(checks.values())
```

#### 2. Performance Metrics
```python
# Track key metrics
metrics = {
    'file_processing_time': measure_processing_time(),
    'flow_execution_time': measure_flow_time(),
    'success_rate': calculate_success_rate(),
    'error_rate': calculate_error_rate()
}
```

### Common Issues and Solutions

#### 1. Flow Execution Errors

**Issue:** Flow not found or not accessible
```
Error: Flow 'xyz' not found in region us-east-1
```
**Solution:**
- Verify flow exists in the correct region
- Check IAM permissions
- Ensure flow is in 'Prepared' status

**Issue:** Invalid input format
```
Error: Input validation failed
```
**Solution:**
- Check input node configuration
- Verify data types match expected format
- Review flow input schema

#### 2. File Processing Errors

**Issue:** PDF text extraction fails
```
Error: Unable to extract text from PDF
```
**Solution:**
- Ensure PDF is not password protected
- Check PDF format compatibility
- Implement fallback OCR processing

**Issue:** Large file handling
```
Error: File size exceeds limit
```
**Solution:**
- Implement file chunking
- Add file size validation
- Use streaming processing for large files

#### 3. Performance Issues

**Issue:** Slow flow execution
```
Warning: Flow execution taking longer than expected
```
**Solution:**
- Optimize prompt templates
- Reduce input data size
- Use appropriate model sizes
- Implement caching strategies

### Debugging Tools

#### 1. Flow Trace Analysis
```python
# Enable detailed tracing
result = client.invoke_flow(
    flow_id=flow_id,
    inputs=inputs,
    enable_trace=True
)

# Analyze trace data
for trace in result.get('trace', []):
    print(f"Node: {trace['node_name']}")
    print(f"Duration: {trace['duration']}")
    print(f"Status: {trace['status']}")
```

#### 2. Debug Mode
```python
# Add debug logging
if debug_mode:
    st.write("Debug Info:")
    st.json({
        'flow_id': flow_id,
        'inputs': inputs,
        'model_config': model_config
    })
```

## 🚀 Deployment Strategies

### Local Development
```bash
# Quick start for development
git clone your-repo
cd bedrock-flows-processor
cp .env.example .env
# Edit .env with your credentials
docker-compose up --build
```

### Staging Environment
```bash
# Deploy to staging with environment variables
docker-compose -f docker-compose.staging.yml up -d
```

### Production Deployment

#### Option 1: AWS ECS
```bash
# Build and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com

docker build -t bedrock-flows-app .
docker tag bedrock-flows-app:latest your-account.dkr.ecr.us-east-1.amazonaws.com/bedrock-flows-app:latest
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/bedrock-flows-app:latest

# Deploy to ECS
aws ecs update-service --cluster your-cluster --service bedrock-flows-service --force-new-deployment
```

#### Option 2: AWS App Runner
```yaml
# apprunner.yaml
version: 1.0
runtime: docker
build:
  commands:
    build:
      - echo "No build commands"
run:
  runtime-version: latest
  command: streamlit run enhanced_streamlit_app.py --server.port=8501
  network:
    port: 8501
    env:
      - name: AWS_DEFAULT_REGION
        value: us-east-1
```

#### Option 3: Kubernetes
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bedrock-flows-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bedrock-flows-app
  template:
    metadata:
      labels:
        app: bedrock-flows-app
    spec:
      containers:
      - name: app
        image: your-registry/bedrock-flows-app:latest
        ports:
        - containerPort: 8501
        env:
        - name: AWS_DEFAULT_REGION
          value: us-east-1
```

## 📊 Cost Optimization

### Usage Monitoring
```python
# Track costs by monitoring usage
def track_usage():
    return {
        'model_invocations': count_model_calls(),
        'tokens_processed': count_tokens(),
        'flow_executions': count_executions(),
        'estimated_cost': calculate_estimated_cost()
    }
```

### Cost Reduction Strategies
1. **Model Selection**: Use appropriate model sizes
2. **Caching**: Cache results for repeated queries
3. **Batching**: Process multiple files together
4. **Optimization**: Reduce token usage with efficient prompts

## 🔐 Security Best Practices

### 1. Credential Management
```bash
# Use AWS Secrets Manager
aws secretsmanager create-secret \
  --name bedrock-flows-credentials \
  --secret-string '{"aws_access_key":"key","aws_secret_key":"secret"}'
```

### 2. Network Security
```yaml
# Use VPC endpoints for Bedrock
VPCEndpoint:
  Type: AWS::EC2::VPCEndpoint
  Properties:
    VpcId: !Ref VPC
    ServiceName: com.amazonaws.us-east-1.bedrock
```

### 3. Data Encryption
```python
# Encrypt sensitive data
from cryptography.fernet import Fernet

def encrypt_file_content(content):
    key = Fernet.generate_key()
    f = Fernet(key)
    encrypted_content = f.encrypt(content.encode())
    return encrypted_content, key
```

## 📈 Scaling Considerations

### Horizontal Scaling
- Use load balancers for multiple app instances
- Implement session management for user state
- Consider microservices architecture for large deployments

### Vertical Scaling
- Increase container resources for CPU/memory intensive operations
- Use GPU instances for image processing workflows
- Optimize memory usage for large file processing

## 🎓 Next Steps

### 1. Advanced Features
- Multi-language support
- Real-time collaboration
- Advanced analytics dashboard
- Custom model fine-tuning

### 2. Integration Options
- REST API for external integrations
- Webhook support for automated workflows
- Database integration for persistent storage
- Third-party service connectors

### 3. Enterprise Features
- Single sign-on (SSO) integration
- Advanced role-based access control
- Comprehensive audit logging
- Custom branding and white-labeling

## 📞 Support and Resources

### Documentation
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Documentation](https://docs.docker.com/)

### Community
- AWS Bedrock Community Forums
- GitHub Issues and Discussions
- Stack Overflow Tags: `aws-bedrock`, `streamlit`

### Professional Support
- AWS Support Plans
- Custom development services
- Training and consultation

---

## 🏁 Quick Reference

### Essential Commands
```bash
# Start application
docker-compose up -d

# View logs
docker-compose logs -f

# Update application
docker-compose pull && docker-compose up -d

# Backup data
docker-compose exec app backup-script.sh

# Health check
curl http://localhost:8501/_stcore/health
```

### Environment Variables
```bash
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
STREAMLIT_SERVER_PORT=8501
DEBUG_MODE=false
```

This completes your comprehensive AWS Bedrock Flows file processing system! 🚀