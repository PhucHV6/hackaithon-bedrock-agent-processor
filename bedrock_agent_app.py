import streamlit as st
import boto3
import json
import pandas as pd
from typing import Dict, Any, Optional, List
from botocore.exceptions import ClientError, NoCredentialsError
import logging
import io
import tempfile
import os
from datetime import datetime
import base64
import PyPDF2
import csv
from io import StringIO
import re
import tiktoken

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedBedrockAgentClient:
    """Enhanced AWS Bedrock Agent client with file processing capabilities"""
    
    def __init__(self, region_name: str = 'us-west-2', aws_access_key_id: str = None, 
                 aws_secret_access_key: str = None):
        """Initialize Bedrock Agent client with optimized timeout settings"""
        try:
            # Configure boto3 with longer timeouts for slow agents
            from botocore.config import Config
            
            config = Config(
                region_name=region_name,
                retries={
                    'max_attempts': 3,
                    'mode': 'adaptive'
                },
                read_timeout=300,  # 5 minutes read timeout
                connect_timeout=60,  # 1 minute connect timeout
                max_pool_connections=50
            )
            
            if aws_access_key_id and aws_secret_access_key:
                self.bedrock_agent = boto3.client(
                    'bedrock-agent',
                    config=config,
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key
                )
                self.bedrock_agent_runtime = boto3.client(
                    'bedrock-agent-runtime',
                    config=config,
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key
                )
            else:
                self.bedrock_agent = boto3.client('bedrock-agent', config=config)
                self.bedrock_agent_runtime = boto3.client('bedrock-agent-runtime', config=config)
            
            self.region_name = region_name
            logger.info(f"Enhanced Bedrock Agent client initialized for region: {region_name}")
            
        except Exception as e:
            logger.error(f"Error initializing Bedrock Agent client: {str(e)}")
            raise

    def test_connection(self) -> bool:
        """Test the connection to AWS Bedrock Agent"""
        try:
            self.bedrock_agent.list_agents(maxResults=1)
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all available agents"""
        try:
            response = self.bedrock_agent.list_agents()
            agents = response.get('agentSummaries', [])
            logger.info(f"Found {len(agents)} agents")
            return agents
        except Exception as e:
            logger.error(f"Error listing agents: {e}")
            st.error(f"Error listing agents: {str(e)}")
            return []

    def get_agent_details(self, agent_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific agent"""
        try:
            response = self.bedrock_agent.get_agent(agentId=agent_id)
            return response.get('agent', {})
        except Exception as e:
            logger.error(f"Error getting agent details: {e}")
            st.error(f"Error getting agent details: {str(e)}")
            return {}

    def list_agent_aliases(self, agent_id: str) -> List[Dict[str, Any]]:
        """List aliases for a specific agent"""
        try:
            response = self.bedrock_agent.list_agent_aliases(agentId=agent_id)
            return response.get('agentAliasSummaries', [])
        except Exception as e:
            logger.error(f"Error listing agent aliases: {e}")
            st.error(f"Error listing agent aliases: {str(e)}")
            return []

    def list_agent_knowledge_bases(self, agent_id: str) -> List[Dict[str, Any]]:
        """List knowledge bases associated with an agent"""
        try:
            response = self.bedrock_agent.list_agent_knowledge_bases(agentId=agent_id)
            return response.get('agentKnowledgeBaseSummaries', [])
        except Exception as e:
            logger.error(f"Error listing agent knowledge bases: {e}")
            return []

    def list_agent_action_groups(self, agent_id: str) -> List[Dict[str, Any]]:
        """List action groups for a specific agent"""
        try:
            response = self.bedrock_agent.list_agent_action_groups(agentId=agent_id)
            return response.get('actionGroupSummaries', [])
        except Exception as e:
            logger.error(f"Error listing agent action groups: {e}")
            return []

    def invoke_agent_interactive(self, agent_id: str, agent_alias_id: str, 
                               session_id: str, input_text: str, 
                               conversation_history: List[Dict] = None) -> Dict[str, Any]:
        """Invoke a Bedrock Agent with conversation context for interactive sessions"""
        try:
            # Build conversation context
            if conversation_history:
                context_parts = ["=== CONVERSATION HISTORY ==="]
                for msg in conversation_history[-5:]:  # Last 5 messages for context
                    role = msg.get('role', 'unknown')
                    content = msg.get('content', '')
                    timestamp = msg.get('timestamp', '')
                    context_parts.append(f"{timestamp} - {role.upper()}: {content}")
                
                context_parts.append("=== CURRENT REQUEST ===")
                context_parts.append(input_text)
                enhanced_input = "\n".join(context_parts)
            else:
                enhanced_input = input_text
            
            # Prepare the invoke request
            invoke_request = {
                'agentId': agent_id,
                'agentAliasId': agent_alias_id,
                'sessionId': session_id,
                'inputText': enhanced_input
            }
            
            logger.info(f"Invoking agent {agent_id} with conversation context")
            response = self.bedrock_agent_runtime.invoke_agent(**invoke_request)
            
            result = self._process_agent_response(response)
            result['conversation_context'] = True
            result['session_id'] = session_id
            return result
            
        except Exception as e:
            logger.error(f"Error invoking interactive agent: {e}")
            return {'error': str(e), 'completion': '', 'completion_reason': 'Error'}

    def invoke_agent_with_file(self, agent_id: str, agent_alias_id: str, 
                             session_id: str, input_text: str, 
                             file_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """Invoke agent with file context"""
        try:
            # Format input with file content if provided
            if file_data:
                content_parts = []
                content_parts.append(f"User Request: {input_text}")
                content_parts.append("=" * 50)
                content_parts.append(f"File Name: {file_data['file_name']}")
                content_parts.append(f"File Type: {file_data['file_type']}")
                
                if file_data.get('metadata'):
                    metadata = file_data['metadata']
                    if metadata.get('word_count'):
                        content_parts.append(f"Word Count: {metadata['word_count']}")
                    if metadata.get('size'):
                        content_parts.append(f"File Size: {metadata['size']} bytes")
                
                content_parts.append("=" * 50)
                content_parts.append("CONTENT:")
                content_parts.append("=" * 50)
                content_parts.append(file_data['file_content'])
                
                formatted_input = "\n".join(content_parts)
            else:
                formatted_input = input_text
            
            # Invoke the agent
            response = self.bedrock_agent_runtime.invoke_agent(
                agentId=agent_id,
                agentAliasId=agent_alias_id,
                sessionId=session_id,
                inputText=formatted_input
            )
            
            return self._process_agent_response(response)
            
        except Exception as e:
            logger.error(f"Error invoking agent with file: {e}")
            return {'error': str(e), 'completion': '', 'completion_reason': 'Error'}

    def _process_agent_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process the streaming response from agent invocation with progress tracking"""
        result = {
            'completion': '',
            'trace': [],
            'completion_reason': None,
            'response_metadata': {},
            'processing_steps': [],
            'citations': [],
            'files': []
        }
        
        try:
            event_stream = response.get('completion', {})
            step_count = 0
            
            # Create a placeholder for progress updates
            progress_placeholder = st.empty()
            status_placeholder = st.empty()
            
            for event in event_stream:
                step_count += 1
                
                # Update progress indicator
                progress_placeholder.info(f"🔄 Processing step {step_count}...")
                
                if 'chunk' in event:
                    chunk = event['chunk']
                    if 'bytes' in chunk:
                        chunk_text = chunk['bytes'].decode('utf-8')
                        result['completion'] += chunk_text
                        status_placeholder.success(f"✅ Received response chunk")
                    
                elif 'trace' in event:
                    trace_event = event['trace']
                    result['trace'].append(trace_event)
                    
                    # Extract citations from trace events with KB metadata
                    if 'knowledgeBaseResponse' in trace_event:
                        kb_response = trace_event['knowledgeBaseResponse']
                        if 'citations' in kb_response:
                            for citation_text in kb_response['citations']:
                                citation = Citation.parse_citation(citation_text, kb_response)
                                result['citations'].append(citation.to_dict())
                    
                    # Extract useful trace info for progress
                    if 'orchestrationTrace' in trace_event:
                        orchestration = trace_event['orchestrationTrace']
                        if 'modelInvocationInput' in orchestration:
                            status_placeholder.info(f"🧠 Processing with LLM...")
                        elif 'observation' in orchestration:
                            status_placeholder.info(f"🔍 Agent observation...")
                    
                elif 'returnControl' in event:
                    # Handle return control events (for function calling)
                    return_control = event['returnControl']
                    result['return_control'] = return_control
                    status_placeholder.warning(f"⏸️ Agent requesting user input")
                
                elif 'files' in event:
                    # Handle file responses
                    files = event['files']
                    result['files'].extend(files)
                    status_placeholder.success(f"📁 Files received")
                
                # Add small delay to show progress (optional)
                import time
                time.sleep(0.05)
            
            # Set completion reason
            if result['completion']:
                result['completion_reason'] = 'SUCCESS'
            elif result.get('return_control'):
                result['completion_reason'] = 'NEEDS_INPUT'
            else:
                result['completion_reason'] = 'NO_OUTPUT'
            
            # Clear progress indicators when done
            progress_placeholder.empty()
            status_placeholder.empty()
            
            result['response_metadata'] = response.get('ResponseMetadata', {})
            result['processing_steps'] = step_count
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing agent response: {e}")
            # Clear progress indicators on error
            if 'progress_placeholder' in locals():
                progress_placeholder.empty()
            if 'status_placeholder' in locals():
                status_placeholder.empty()
            raise

    def analyze_agent_response_for_questions(self, completion_text: str) -> Dict[str, Any]:
        """Analyze agent response to detect if agent is asking questions"""
        agent_questions = []
        requires_input = False
        
        if completion_text:
            # Look for question patterns in agent responses
            question_indicators = [
                "?", "please provide", "can you tell me", "what is your", 
                "which", "how many", "do you have", "clarification needed",
                "need more information", "could you specify", "please clarify"
            ]
            
            content_lower = completion_text.lower()
            if any(indicator in content_lower for indicator in question_indicators):
                # Extract potential questions
                sentences = completion_text.split('.')
                questions = [s.strip() + '?' for s in sentences if '?' in s or any(ind in s.lower() for ind in question_indicators[:5])]
                
                if questions:
                    agent_questions.extend([{
                        'question': q,
                        'context': {'full_response': completion_text}
                    } for q in questions])
                    requires_input = True
        
        return {
            'agent_questions': agent_questions,
            'requires_input': requires_input
        }

class FileProcessor:
    """Handle different file types and extract content"""
    
    @staticmethod
    def process_text_file(file_content: bytes, file_name: str) -> Dict[str, Any]:
        """Process plain text files"""
        try:
            content = file_content.decode('utf-8')
            return {
                'file_type': 'text',
                'file_name': file_name,
                'file_content': content,
                'metadata': {
                    'word_count': len(content.split()),
                    'char_count': len(content),
                    'line_count': len(content.splitlines()),
                    'size': len(file_content)
                }
            }
        except Exception as e:
            st.error(f"Error processing text file: {e}")
            return None

    @staticmethod
    def process_pdf_file(file_content: bytes, file_name: str) -> Dict[str, Any]:
        """Process PDF files"""
        try:
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            text_content = ""
            
            for page in pdf_reader.pages:
                text_content += page.extract_text() + "\n"
            
            return {
                'file_type': 'pdf',
                'file_name': file_name,
                'file_content': text_content,
                'metadata': {
                    'page_count': len(pdf_reader.pages),
                    'word_count': len(text_content.split()),
                    'char_count': len(text_content),
                    'size': len(file_content)
                }
            }
        except Exception as e:
            st.error(f"Error processing PDF file: {e}")
            return None

    @staticmethod
    def process_json_file(file_content: bytes, file_name: str) -> Dict[str, Any]:
        """Process JSON files"""
        try:
            content = file_content.decode('utf-8')
            json_data = json.loads(content)
            
            return {
                'file_type': 'json',
                'file_name': file_name,
                'file_content': content,
                'json_data': json_data,
                'metadata': {
                    'structure': FileProcessor._analyze_json_structure(json_data),
                    'size': len(content)
                }
            }
        except Exception as e:
            st.error(f"Error processing JSON file: {e}")
            return None

    @staticmethod
    def process_csv_file(file_content: bytes, file_name: str) -> Dict[str, Any]:
        """Process CSV files"""
        try:
            content = file_content.decode('utf-8')
            csv_file = StringIO(content)
            reader = csv.reader(csv_file)
            
            rows = list(reader)
            headers = rows[0] if rows else []
            data_rows = rows[1:] if len(rows) > 1 else []
            
            return {
                'file_type': 'csv',
                'file_name': file_name,
                'file_content': content,
                'metadata': {
                    'headers': headers,
                    'row_count': len(data_rows),
                    'column_count': len(headers),
                    'preview': data_rows[:5],  # First 5 rows
                    'size': len(file_content)
                }
            }
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
            return None

    @staticmethod
    def process_markdown_file(file_content: bytes, file_name: str) -> Dict[str, Any]:
        """Process Markdown files"""
        try:
            content = file_content.decode('utf-8')
            
            # Basic markdown analysis
            lines = content.splitlines()
            headers = [line for line in lines if line.startswith('#')]
            
            return {
                'file_type': 'markdown',
                'file_name': file_name,
                'file_content': content,
                'metadata': {
                    'word_count': len(content.split()),
                    'line_count': len(lines),
                    'header_count': len(headers),
                    'headers': headers[:10],  # First 10 headers
                    'size': len(file_content)
                }
            }
        except Exception as e:
            st.error(f"Error processing Markdown file: {e}")
            return None

    @staticmethod
    def _analyze_json_structure(data: Any, depth: int = 0) -> str:
        """Analyze JSON structure recursively"""
        if depth > 3:  # Limit recursion depth
            return "..."
        
        if isinstance(data, dict):
            keys = list(data.keys())[:5]  # Show first 5 keys
            return f"Object with {len(data)} keys: {keys}"
        elif isinstance(data, list):
            if data:
                return f"Array with {len(data)} items of type: {type(data[0]).__name__}"
            else:
                return "Empty array"
        else:
            return f"{type(data).__name__}: {str(data)[:50]}"

class Citation:
    """Represents a citation from an agent response"""
    def __init__(self, text: str, source: Optional[str] = None, 
                 page: Optional[int] = None, confidence: Optional[float] = None,
                 kb_id: Optional[str] = None, 
                 data_source_name: Optional[str] = None,
                 data_source_type: Optional[str] = None,
                 vector_score: Optional[float] = None):
        self.text = text
        self.source = source
        self.confidence = confidence
        self.page = page
        self.timestamp = datetime.now()
        # Bedrock Knowledge Base specific fields
        self.kb_id = kb_id
        self.data_source_name = data_source_name
        self.data_source_type = data_source_type
        self.vector_score = vector_score

    def to_dict(self) -> Dict[str, Any]:
        """Convert citation to dictionary format"""
        return {
            'text': self.text,
            'source': self.source,
            'page': self.page,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'kb_id': self.kb_id,
            'data_source_name': self.data_source_name,
            'data_source_type': self.data_source_type,
            'vector_score': self.vector_score
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Citation':
        """Create Citation object from dictionary"""
        citation = Citation(
            text=data['text'],
            source=data.get('source'),
            page=data.get('page'),
            confidence=data.get('confidence'),
            kb_id=data.get('kb_id'),
            data_source_name=data.get('data_source_name'),
            data_source_type=data.get('data_source_type'),
            vector_score=data.get('vector_score')
        )
        if 'timestamp' in data:
            citation.timestamp = datetime.fromisoformat(data['timestamp'])
        return citation

    @staticmethod
    def parse_citation(citation_text: str, kb_response: Optional[Dict[str, Any]] = None) -> 'Citation':
        """Parse citation text and knowledge base response to extract metadata"""
        # Initialize citation with original text
        citation = Citation(citation_text)
        
        try:
            # Look for source in parentheses or brackets
            source_match = re.search(r'[\(\[](.*?)[\)\]]', citation_text)
            if source_match:
                citation.source = source_match.group(1)
            
            # Look for page numbers
            page_match = re.search(r'page[s]?\s*(\d+)', citation_text.lower())
            if page_match:
                citation.page = int(page_match.group(1))
            
            # Look for confidence scores
            conf_match = re.search(r'confidence:\s*([\d.]+)', citation_text.lower())
            if conf_match:
                citation.confidence = float(conf_match.group(1))
            
            # Extract Bedrock Knowledge Base metadata if available
            if kb_response:
                # Extract knowledge base ID if present
                if 'knowledgeBaseId' in kb_response:
                    citation.kb_id = kb_response['knowledgeBaseId']
                
                # Extract data source information
                if 'retrievalResults' in kb_response:
                    for result in kb_response['retrievalResults']:
                        if 'location' in result:
                            location = result['location']
                            citation.data_source_name = location.get('s3Location', {}).get('uri') or location.get('dataSourceName')
                            citation.data_source_type = location.get('type')
                        if 'score' in result:
                            citation.vector_score = float(result['score'])
                            # Use vector score as confidence if no explicit confidence was found
                            if citation.confidence is None:
                                citation.confidence = citation.vector_score
                
        except Exception as e:
            logger.warning(f"Error parsing citation: {e}")
            
        return citation

def init_session_state():
    """Initialize Streamlit session state"""
    if 'bedrock_client' not in st.session_state:
        st.session_state.bedrock_client = None
    if 'agents' not in st.session_state:
        st.session_state.agents = []
    if 'selected_agent' not in st.session_state:
        st.session_state.selected_agent = None
    if 'agent_aliases' not in st.session_state:
        st.session_state.agent_aliases = []
    if 'execution_history' not in st.session_state:
        st.session_state.execution_history = []
    if 'processed_file' not in st.session_state:
        st.session_state.processed_file = None
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'chat_input_key' not in st.session_state:
        st.session_state.chat_input_key = 0
    if 'chat_session_id' not in st.session_state:
        import uuid
        st.session_state.chat_session_id = str(uuid.uuid4())
    if 'processing_message' not in st.session_state:
        st.session_state.processing_message = None
    if 'needs_processing' not in st.session_state:
        st.session_state.needs_processing = False
    if 'tiktoken_enc' not in st.session_state:
        try:
            st.session_state.tiktoken_enc = tiktoken.get_encoding("cl100k_base")
        except:
            st.session_state.tiktoken_enc = None

def setup_sidebar():
    """Setup sidebar for AWS configuration and navigation"""
    if 'is_logged_in' not in st.session_state:
        st.session_state.is_logged_in = False

    # Initialize selected page in session state if not exists
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = "🤖 Agents"

    def on_page_change():
        st.session_state.selected_page = st.session_state.nav_radio

    if not st.session_state.is_logged_in:
        st.sidebar.title("🔧 AWS Configuration")
        
        # AWS Region
        region = st.sidebar.selectbox(
            "AWS Region",
            ["us-west-2", "us-east-1", "eu-west-1", "ap-southeast-1"],
            index=0,
            help="Select the region where your Bedrock Agent is deployed"
        )
        
        # Credentials input
        st.sidebar.subheader("AWS Credentials")
        cred_method = st.sidebar.radio(
            "Authentication Method",
            ["Use Default Credentials", "Enter Credentials Manually"]
        )
        
        aws_access_key = None
        aws_secret_key = None
        
        if cred_method == "Enter Credentials Manually":
            aws_access_key = st.sidebar.text_input("AWS Access Key ID", type="password")
            aws_secret_key = st.sidebar.text_input("AWS Secret Access Key", type="password")
        
        # Connect button
        if st.sidebar.button("🔌 Connect to AWS"):
            try:
                with st.spinner("Connecting to AWS Bedrock Agents..."):
                    client = EnhancedBedrockAgentClient(
                        region_name=region,
                        aws_access_key_id=aws_access_key,
                        aws_secret_access_key=aws_secret_key
                    )
                    
                    # Test the connection by listing agents
                    agents = client.list_agents()
                    
                    st.session_state.bedrock_client = client
                    st.session_state.agents = agents
                    st.session_state.is_logged_in = True

                    # Only set initial agent and alias if none selected
                    if not st.session_state.selected_agent and agents:
                        first_agent = agents[0]
                        aliases = client.list_agent_aliases(first_agent['agentId'])
                        if aliases:
                            # Sort aliases by creation date (newest first)
                            sorted_aliases = sorted(aliases, 
                                key=lambda x: x.get('lastUpdatedDateTime', x.get('creationDateTime', '')), 
                                reverse=True)
                            st.session_state.chat_alias_id = sorted_aliases[0]['agentAliasId']
                            st.session_state.selected_agent = first_agent
                    
                    st.sidebar.success("✅ Connected successfully!")
                    st.rerun()
                
            except Exception as e:
                error_msg = str(e)
                st.sidebar.error(f"❌ Connection failed: {error_msg}")
                
                # Provide helpful debugging information
                if "credentials" in error_msg.lower():
                    st.sidebar.info("💡 Tip: Check your AWS credentials and permissions")
                elif "region" in error_msg.lower():
                    st.sidebar.info("💡 Tip: Verify the selected region supports Bedrock Agents")
                elif "access denied" in error_msg.lower():
                    st.sidebar.info("💡 Tip: Ensure your IAM user/role has Bedrock permissions")
    else:
        # Show current connection info
        st.sidebar.success(f"✅ Connected to {st.session_state.bedrock_client.region_name}")
        
        # Navigation menu
        st.sidebar.title("🧭 Navigation")
        
        # Navigation radio buttons with custom styling
        st.markdown("""
            <style>
            /* Style for radio buttons in sidebar */
            .stRadio [role="radiogroup"] {
                gap: 0.5rem;
            }
            .stRadio [data-testid="stMarkdownContainer"] > p {
                font-size: 1rem;
                margin: 0;
            }
            </style>
        """, unsafe_allow_html=True)
        
        # Use on_change callback for the radio button
        st.sidebar.radio(
            "Go to",
            ["🤖 Agents", "💬 Chat", "📈 History"],
            key="nav_radio",
            on_change=on_page_change,
            index=["🤖 Agents", "💬 Chat", "📈 History"].index(st.session_state.selected_page)
        )
        
        # Show agent info if selected
        if st.session_state.selected_agent:
            st.sidebar.success(f"🤖 Selected Agent: {st.session_state.selected_agent['agentName']}")
        
        # Logout button at bottom of sidebar
        st.sidebar.markdown("---")
        if st.sidebar.button("🚪 Logout"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        # Debug section (expandable)
        with st.sidebar.expander("🔍 Debug Information"):
            st.write("**Connection Status:** ✅ Connected")
            st.write(f"**Region:** {st.session_state.bedrock_client.region_name}")
            
            if st.button("Test Connection"):
                if st.session_state.bedrock_client.test_connection():
                    st.success("Connection test passed!")
                else:
                    st.error("Connection test failed!")
    
    return st.session_state.is_logged_in

def display_agents_section():
    """Display available agents section with detailed agent analysis"""
    st.header("🤖 Available Agents")
    
    if not st.session_state.agents:
        st.info("No agents found. Please ensure you have agents created in AWS Bedrock.")
        return
    
    # Create agent selection
    agent_options = [f"{agent['agentName']} ({agent['agentId']})" for agent in st.session_state.agents]
    
    # Find current agent index
    current_index = 0
    if st.session_state.selected_agent:
        current_index = next(
            (i for i, opt in enumerate(agent_options) 
             if st.session_state.selected_agent['agentId'] in opt),
            0
        )
    
    # Track previous selection
    if 'previous_agent_selection' not in st.session_state:
        st.session_state.previous_agent_selection = agent_options[current_index]
    
    selected_agent_option = st.selectbox(
        "Select an Agent",
        agent_options,
        index=current_index,
        key="agent_selection"
    )
    
    # Only update if selection changed
    if selected_agent_option != st.session_state.previous_agent_selection:
        st.session_state.previous_agent_selection = selected_agent_option
        # Extract agent ID from selection
        agent_id = selected_agent_option.split('(')[-1].strip(')')
        selected_agent = next(agent for agent in st.session_state.agents if agent['agentId'] == agent_id)
        previous_agent_name = st.session_state.selected_agent['agentName'] if st.session_state.selected_agent else None
        st.session_state.selected_agent = selected_agent
        
        # Add context message about agent change
        if previous_agent_name and st.session_state.chat_messages:
            context_msg = {
                "role": "system",
                "content": f"Note: Switching from agent '{previous_agent_name}' to '{selected_agent['agentName']}'. Previous conversation context is maintained.",
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "is_context": True
            }
            st.session_state.chat_messages.append(context_msg)
        
        st.rerun()

    if selected_agent_option:
        # Extract agent ID from selection
        agent_id = selected_agent_option.split('(')[-1].strip(')')
        if not st.session_state.selected_agent or st.session_state.selected_agent['agentId'] != agent_id:
            selected_agent = next(agent for agent in st.session_state.agents if agent['agentId'] == agent_id)
            st.session_state.selected_agent = selected_agent
        agent_name = st.session_state.selected_agent['agentName']

        # Load agent aliases and select newest by default
        try:
            aliases = st.session_state.bedrock_client.list_agent_aliases(agent_id)
            st.session_state.agent_aliases = aliases
            
            if aliases:
                # Sort aliases by creation date (newest first)
                sorted_aliases = sorted(aliases, 
                    key=lambda x: x.get('lastUpdatedDateTime', x.get('creationDateTime', '')), 
                    reverse=True)
                newest_alias = sorted_aliases[0]
                
                # Initialize use_latest in session state if not present
                if 'use_latest_alias' not in st.session_state:
                    st.session_state.use_latest_alias = True
                    st.session_state.chat_alias_id = newest_alias['agentAliasId']
                
                # Add toggle for testing different aliases
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info(f"🔄 Latest alias: {newest_alias['agentAliasName']}")
                with col2:
                    # Handle toggle state change
                    previous_state = st.session_state.use_latest_alias
                    current_state = st.toggle(
                        "Use Latest Alias",
                        value=previous_state,
                        help="Toggle to test different aliases",
                        key="use_latest_toggle"
                    )
                    
                    # If state changed, update and rerun
                    if current_state != previous_state:
                        st.session_state.use_latest_alias = current_state
                        if current_state:  # If switching to latest
                            st.session_state.chat_alias_id = newest_alias['agentAliasId']
                        st.rerun()
                
                if st.session_state.use_latest_alias:
                    if st.session_state.chat_alias_id != newest_alias['agentAliasId']:
                        st.session_state.chat_alias_id = newest_alias['agentAliasId']
                else:
                    alias_options = [f"{alias['agentAliasName']} ({alias['agentAliasId']})" for alias in aliases]
                    # Find current index for the selectbox
                    current_index = 0
                    if st.session_state.chat_alias_id:
                        current_index = next(
                            (i for i, opt in enumerate(alias_options) 
                             if st.session_state.chat_alias_id in opt),
                            0
                        )
                    
                    # Handle alias selection change
                    if 'previous_alias_selection' not in st.session_state:
                        st.session_state.previous_alias_selection = alias_options[current_index]
                    
                    selected_alias_option = st.selectbox(
                        "Select an alias for testing:",
                        alias_options,
                        index=current_index,
                        key="agent_section_alias"
                    )
                    
                    # If selection changed, update and rerun
                    if selected_alias_option != st.session_state.previous_alias_selection:
                        st.session_state.previous_alias_selection = selected_alias_option
                        selected_alias_id = selected_alias_option.split('(')[-1].strip(')')
                        if st.session_state.chat_alias_id != selected_alias_id:
                            st.session_state.chat_alias_id = selected_alias_id
                            st.rerun()
            else:
                st.warning("No aliases available. Using version identifiers.")
                if 'use_version' not in st.session_state:
                    st.session_state.use_version = True
                    st.session_state.chat_alias_id = "DRAFT"
                
                # Handle version toggle state change
                previous_version_state = st.session_state.use_version
                current_version_state = st.checkbox(
                    "Use agent version instead of alias",
                    value=previous_version_state,
                    key="use_version_toggle"
                )
                
                # If version toggle state changed, update and rerun
                if current_version_state != previous_version_state:
                    st.session_state.use_version = current_version_state
                    if current_version_state:
                        st.session_state.chat_alias_id = "DRAFT"
                    else:
                        st.session_state.chat_alias_id = None
                    st.rerun()
                
                if st.session_state.use_version:
                    version_input = st.text_input(
                        "Version:",
                        value=st.session_state.chat_alias_id or "DRAFT",
                        key="chat_version"
                    )
                    if st.session_state.chat_alias_id != version_input:
                        st.session_state.chat_alias_id = version_input
                        st.rerun()
                else:
                    st.session_state.chat_alias_id = None
                
        except Exception as e:
            st.error(f"Error loading agent aliases: {str(e)}")
        
        # Display detailed agent information
        try:
            # Get detailed agent information
            agent_details = st.session_state.bedrock_client.get_agent_details(agent_id)
            
            # Display detailed information
            display_agent_details(agent_name, agent_details)
            
        except Exception as e:
            st.error(f"Error fetching agent details: {str(e)}")
            with st.expander(f"ℹ️ Basic Agent Information"):
                st.markdown(f"""
                **Agent Name:** {agent_name}  
                **Agent ID:** `{agent_id}`
                
                ⚠️ Unable to fetch detailed information. Please check your permissions and try again.
                """)

def display_agent_details(agent_name: str, agent_details: Dict[str, Any]):
    """Display detailed agent information"""
    capabilities = []
    
    # Basic agent information
    agent_id = agent_details.get('agentId', 'Unknown')
    agent_arn = agent_details.get('agentArn', 'Unknown')
    agent_status = agent_details.get('agentStatus', 'Unknown')
    creation_date = agent_details.get('creationDateTime', None)
    last_update = agent_details.get('lastUpdatedDateTime', None)
    
    # Check for capabilities
    if agent_details.get('instruction'):
        capabilities.append("📋 Custom instructions")
    if agent_details.get('foundationModel'):
        capabilities.append("🧠 Foundation Model powered")
    
    # Create configuration tab
    st.markdown("### 📌 Agent Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Agent Configuration**")
        st.write(f"- 🆔 ID: `{agent_id}`")
        status_color = "green" if agent_status == "Active" else "orange"
        st.markdown(f"- 📊 Status: <span style='color: {status_color}'>{agent_status}</span>", unsafe_allow_html=True)
        if agent_details.get('foundationModel'):
            st.write(f"- 🧠 Model: `{agent_details['foundationModel']}`")
        if agent_details.get('inferenceConfiguration'):
            inference = agent_details['inferenceConfiguration']
            if inference.get('temperature'):
                st.write(f"- 🌡️ Temperature: {inference['temperature']}")
            if inference.get('topP'):
                st.write(f"- 📊 Top P: {inference['topP']}")
            if inference.get('maxTokens'):
                st.write(f"- 📝 Max Tokens: {inference['maxTokens']}")
    
    with col2:
        st.markdown("**Timestamps**")
        if creation_date:
            st.write(f"- 📅 Created: {creation_date.strftime('%Y-%m-%d %H:%M:%S UTC') if isinstance(creation_date, datetime) else creation_date}")
        if last_update:
            st.write(f"- 🔄 Last Updated: {last_update.strftime('%Y-%m-%d %H:%M:%S UTC') if isinstance(last_update, datetime) else last_update}")
        if agent_details.get('idleSessionTTLInSeconds'):
            st.write(f"- ⏱️ Session TTL: {agent_details['idleSessionTTLInSeconds']}s")

    if capabilities:
        st.markdown("**🚀 Capabilities**")
        for capability in capabilities:
            st.write(f"- {capability}")
    
    if agent_details.get('instruction'):
        st.markdown("**📝 Agent Instructions**")
        st.code(agent_details['instruction'], language="markdown")

def calculate_file_tokens(file_content: str, file_type: str = None) -> int:
    """Calculate accurate token count using tiktoken
    
    Uses OpenAI's tiktoken library for precise token counting, which is the same 
    tokenizer used by the actual models.
    
    Args:
        file_content: The text content to analyze
        file_type: Optional file extension (not used since we use model's tokenizer)
    
    Returns:
        Actual token count as would be seen by the model
    """
    if not file_content:
        return 0
        
    try:
        # Use cl100k_base encoding (used by gpt-4, gpt-3.5-turbo, text-embedding-ada-002)
        enc = tiktoken.get_encoding("cl100k_base")
        # Get actual token count
        tokens = enc.encode(file_content)
        return len(tokens)
    except Exception as e:
        # Fallback to approximate counting if tiktoken fails
        logger.warning(f"Tiktoken encoding failed: {str(e)}. Using fallback estimation.")
        # Fallback estimation
        words = len(file_content.split())
        return int(words * 1.3)  # Rough estimate

def execute_smart_chat_agent(user_message: str, file_data: Optional[Dict[str, Any]] = None):
    """Execute agent with smart conversation context"""
    try:
        # Generate session ID if not exists
        if 'chat_session_id' not in st.session_state:
            import uuid
            st.session_state.chat_session_id = str(uuid.uuid4())
        
        # Create containers for progress tracking
        progress_container = st.container()
        
        # Execute agent with conversation context
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_start = datetime.now()
            
            status_text.info("🚀 Starting smart conversation with agent...")
            progress_bar.progress(10)
            
            status_text.info("🤖 Processing with conversation context...")
            progress_bar.progress(25)
            
            # Pass conversation history for context
            conversation_history = st.session_state.chat_messages  # Include current message
            
            # Manage conversation history length
            MAX_MESSAGES = 20  # Adjust based on typical message length
            if len(conversation_history) > MAX_MESSAGES:
                # Keep the first message (could be file context) and most recent messages
                conversation_history = [conversation_history[0]] + conversation_history[-MAX_MESSAGES:]
                summary_msg = {
                    "role": "system",
                    "content": f"Note: The conversation is long. Showing initial context and the last {MAX_MESSAGES} messages.",
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "is_context": True
                }
                conversation_history.insert(1, summary_msg)
            
            # Add agent context to the input
            agent_context = (
                f"Previous conversation history is available. "
                f"You are agent '{st.session_state.selected_agent['agentName']}'. "
                f"Please review the conversation history and continue assisting based on the context."
            )
            
            # Calculate approximate token count for context management
            total_text = "\n".join([msg.get("content", "") for msg in conversation_history])
            approx_tokens = len(total_text.split()) * 1.3  # Rough estimation
            
            if approx_tokens > 150000:  # Leave room for response
                st.warning("⚠️ Conversation is very long. Summarizing history to maintain performance.")
                # Keep essential context
                conversation_history = [
                    conversation_history[0],  # First message (might be file context)
                    {
                        "role": "system",
                        "content": "Note: The conversation history has been summarized due to length.",
                        "is_context": True
                    },
                    *conversation_history[-10:]  # Last 10 messages
                ]
            
            enhanced_input = f"{agent_context}\n\nUser's message: {user_message}"
            
            if file_data:
                # First message with file
                result = st.session_state.bedrock_client.invoke_agent_with_file(
                    agent_id=st.session_state.selected_agent['agentId'],
                    agent_alias_id=st.session_state.chat_alias_id,
                    session_id=st.session_state.chat_session_id,
                    input_text=enhanced_input,
                    file_data=file_data
                )
            else:
                # Regular conversation or follow-up
                result = st.session_state.bedrock_client.invoke_agent_interactive(
                    agent_id=st.session_state.selected_agent['agentId'],
                    agent_alias_id=st.session_state.chat_alias_id,
                    session_id=st.session_state.chat_session_id,
                    input_text=enhanced_input,
                    conversation_history=conversation_history
                )
            
            # Final progress
            progress_bar.progress(100)
            execution_time = (datetime.now() - time_start).total_seconds()
            
            # Analyze result for questions
            if result.get('completion'):
                question_analysis = st.session_state.bedrock_client.analyze_agent_response_for_questions(result['completion'])
                
                if question_analysis.get('requires_input'):
                    status_text.warning(f"🤖 Agent has questions after {execution_time:.1f} seconds")
                else:
                    status_text.success(f"✅ Agent completed in {execution_time:.1f} seconds")
            else:
                status_text.success(f"✅ Agent completed in {execution_time:.1f} seconds")
            
            # Clear progress after a moment
            import time
            time.sleep(2)
            progress_bar.empty()
            status_text.empty()
            
            # Clear uploaded files after getting response
            st.session_state.uploaded_files = {}
            st.session_state.total_tokens = 0
            st.session_state.file_uploader_key += 1  # Force file uploader to reset
        
        # Get alias name
        if st.session_state.agent_aliases:
            alias_name = next((alias['agentAliasName'] for alias in st.session_state.agent_aliases 
                              if alias['agentAliasId'] == st.session_state.chat_alias_id), f'Version: {st.session_state.chat_alias_id}')
        else:
            alias_name = f'Version: {st.session_state.chat_alias_id}'
        
        # Analyze for detected questions
        detected_questions = []
        if result.get('completion'):
            question_analysis = st.session_state.bedrock_client.analyze_agent_response_for_questions(result['completion'])
            detected_questions = question_analysis.get('agent_questions', [])
        
        # Add assistant response to chat
        assistant_msg = {
            "role": "assistant",
            "content": "Here's my analysis:" if not detected_questions else "I have some questions to better help you:",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "agent_result": result,
            "alias_name": alias_name,
            "file_name": file_data['file_name'] if file_data else None,
            "execution_time": execution_time,
            "detected_questions": detected_questions
        }
        st.session_state.chat_messages.append(assistant_msg)
        
        # Save conversation to history
        if 'execution_history' not in st.session_state:
            st.session_state.execution_history = []
            
        # Create or update the history entry
        if not st.session_state.execution_history:
            # First entry in history
            history_entry = {
                'timestamp_start': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'timestamp_last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'session_id': st.session_state.chat_session_id,
                'conversation': st.session_state.chat_messages.copy(),
                'agents_used': [{
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'agent_name': st.session_state.selected_agent['agentName'],
                    'agent_id': st.session_state.selected_agent['agentId'],
                    'alias_name': alias_name
                }],
                'files_processed': [],
                'execution_times': []
            }
            if file_data:
                history_entry['files_processed'].append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'file_info': file_data['file_name'],
                    'file_type': file_data['file_type']
                })
            history_entry['execution_times'].append(execution_time)
            st.session_state.execution_history.append(history_entry)
        else:
            # Update the last history entry
            current_entry = st.session_state.execution_history[-1]
            current_entry['timestamp_last_update'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            current_entry['conversation'] = st.session_state.chat_messages.copy()
            
            # Add new agent if different from the last one used
            last_agent = current_entry['agents_used'][-1]
            if (last_agent['agent_id'] != st.session_state.selected_agent['agentId'] or 
                last_agent['alias_name'] != alias_name):
                current_entry['agents_used'].append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'agent_name': st.session_state.selected_agent['agentName'],
                    'agent_id': st.session_state.selected_agent['agentId'],
                    'alias_name': alias_name
                })
            
            # Add file information if present
            if file_data:
                current_entry['files_processed'].append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'file_info': file_data['file_name'],
                    'file_type': file_data['file_type']
                })
            current_entry['execution_times'].append(execution_time)
            
        # Clear uploaded files even on error
        st.session_state.uploaded_files = {}
        st.session_state.total_tokens = 0
        st.session_state.file_uploader_key += 1  # Force file uploader to reset

    except Exception as e:
        error_msg = str(e)
        st.error(f"❌ Error executing smart conversation with agent: {error_msg}")
        
        # Enhanced error handling
        if "timeout" in error_msg.lower():
            st.warning("⏱️ **Smart Conversation Timeout**")
            st.info("""
            **Conversation with agents may take longer due to:**
            - Context analysis from previous messages
            - Knowledge base retrieval
            - Action group execution
            - Complex multi-step reasoning
            
            **Try:**
            - Break down complex requests
            - Provide specific information upfront
            - Use shorter, more focused messages
            """)
        
        elif "credentials" in error_msg.lower():
            st.info("🔑 **Credentials Issue**: Check your AWS credentials and permissions")
        
        elif "access denied" in error_msg.lower():
            st.info("🚫 **Permission Issue**: Verify IAM permissions for Bedrock Agents")
        
        # Add error message to chat
        error_chat_msg = {
            "role": "assistant",
            "content": f"Sorry, I encountered an error: {error_msg}",
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "is_error": True,
            "error_details": {
                "error_type": "timeout" if "timeout" in error_msg.lower() else "other",
                "error_message": error_msg,
                "session_id": st.session_state.chat_session_id
            }
        }
        st.session_state.chat_messages.append(error_chat_msg)
        
        # Clear uploaded files even on error
        st.session_state.uploaded_files = {}
        st.session_state.total_tokens = 0
        st.session_state.file_uploader_key += 1  # Force file uploader to reset

def display_smart_agent_result(agent_result: Dict[str, Any]):
    """Display agent results with intelligent question detection"""
    if not agent_result:
        st.error("No response received from agent")
        return
    
    if agent_result.get('error'):
        st.error(f"Agent execution error: {agent_result['error']}")
        return
    
    completion = agent_result.get('completion', '')
    if not completion:
        st.warning("Agent completed but returned no response.")
        with st.expander("🔍 Debug Info - Raw Agent Result"):
            st.json(agent_result)
        return
    
    # Analyze response for agent questions
    if st.session_state.bedrock_client:
        question_analysis = st.session_state.bedrock_client.analyze_agent_response_for_questions(completion)
        
        if question_analysis.get('requires_input'):
            st.session_state.waiting_for_user_input = True
            st.session_state.detected_agent_questions = question_analysis.get('agent_questions', [])
        else:
            st.session_state.waiting_for_user_input = False
            st.session_state.detected_agent_questions = []
    
    # Display the main agent response
    st.markdown("#### 🤖 Agent Response")
    display_content_with_formatting(completion)
    
    # Show citations if available
    if agent_result.get('citations'):
        st.markdown("### 📚 Citations")
        
        # Group citations by knowledge base and source
        citations_by_kb = {}
        for citation_data in agent_result['citations']:
            citation = Citation.from_dict(citation_data)
            kb_id = citation.kb_id or "Unknown KB"
            source = citation.data_source_name or citation.source or "Unknown Source"
            
            if kb_id not in citations_by_kb:
                citations_by_kb[kb_id] = {}
            if source not in citations_by_kb[kb_id]:
                citations_by_kb[kb_id][source] = []
            citations_by_kb[kb_id][source].append(citation)
        
        # Display citations grouped by knowledge base and source
        for kb_id, sources in citations_by_kb.items():
            st.subheader(f"Knowledge Base: {kb_id}")
            
            for source, citations in sources.items():
                with st.expander(f"📖 {source} ({len(citations)} citations)"):
                    for i, citation in enumerate(citations, 1):
                        st.markdown(f"**Citation {i}:**")
                        st.write(citation.text)
                        
                        # Show metadata if available
                        metadata_rows = []
                        
                        # Row 1: Source information
                        source_info = []
                        if citation.data_source_type:
                            source_info.append(f"📁 Type: {citation.data_source_type}")
                        if citation.page is not None:
                            source_info.append(f"📄 Page: {citation.page}")
                        if source_info:
                            metadata_rows.append(" | ".join(source_info))
                        
                        # Row 2: Relevance scores
                        score_info = []
                        if citation.vector_score is not None:
                            score_color = "green" if citation.vector_score > 0.7 else "orange"
                            score_info.append(f"🎯 Vector Score: <span style='color: {score_color}'>{citation.vector_score:.3f}</span>")
                        if citation.confidence is not None and citation.confidence != citation.vector_score:
                            conf_color = "green" if citation.confidence > 0.8 else "orange"
                            score_info.append(f"✨ Confidence: <span style='color: {conf_color}'>{citation.confidence:.2f}</span>")
                        if score_info:
                            metadata_rows.append(" | ".join(score_info))
                        
                        # Row 3: Timestamp
                        if citation.timestamp:
                            metadata_rows.append(f"🕒 Retrieved: {citation.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                        
                        # Display metadata rows
                        for row in metadata_rows:
                            st.markdown(row, unsafe_allow_html=True)
                        
                        st.markdown("---")
    
    # Show files if any were returned
    if agent_result.get('files'):
        st.markdown("### 📁 Files")
        for file_info in agent_result['files']:
            st.write(f"**File:** {file_info}")
    
    # Show execution status
    completion_reason = agent_result.get('completion_reason')
    if completion_reason:
        if completion_reason == 'SUCCESS':
            if st.session_state.waiting_for_user_input:
                st.info("🤖 Agent completed with questions - please respond to continue")
            else:
                st.success("✅ Agent completed successfully")
        else:
            st.warning(f"⚠️ Agent status: {completion_reason}")
    
    # Show execution time if available
    execution_time = agent_result.get('execution_time')
    if execution_time:
        st.caption(f"⏱️ Execution time: {execution_time:.1f} seconds")
    
    # Show debug information if trace is available
    if agent_result.get('trace') and len(agent_result['trace']) > 0:
        with st.expander("🔍 Execution Trace"):
            for i, trace_event in enumerate(agent_result['trace']):
                st.write(f"**Trace Event {i+1}:**")
                st.json(trace_event)

def get_chat_style():
    return """
    <style>
    /* Modern chat container */
    .chat-container {
        max-width: 900px;
        margin: 0 auto;
        padding: 20px;
        padding-bottom: 100px;  /* Add padding for input area */
    }
    
    /* Input container at bottom but not fixed */
    .input-container {
        background: white;
        padding: 20px 40px;
        border-top: 1px solid #e5e5e5;
        margin-top: 20px;
    }
    
    /* Modern input area */
    .stTextArea textarea {
        border-radius: 12px !important;
        border: 1px solid #e5e5e5 !important;
        padding: 15px !important;
        font-size: 16px !important;
        line-height: 1.5 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
        resize: none !important;
        min-height: 100px !important;
    }

    /* Style expander */
    .streamlit-expanderHeader {
        font-size: 14px !important;
        padding: 8px 12px !important;
        background: none !important;
        border: none !important;
        color: #6c757d !important;
        margin-top: 8px !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: #f8f9fa !important;
        border-radius: 8px !important;
    }

    .streamlit-expanderContent {
        border: none !important;
        padding: 10px 0 !important;
    }

    /* File uploader in expander */
    .stFileUploader {
        width: 100% !important;
        padding: 0 !important;
    }

    .stFileUploader > div {
        padding: 0 !important;
    }

    /* File attachment indicator */
    .file-indicator {
        display: inline-flex;
        align-items: center;
        padding: 4px 8px;
        background: #f1f3f5;
        border-radius: 12px;
        font-size: 12px;
        color: #495057;
        margin-right: 8px;
    }
    
    /* Send button */
    .stButton button {
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
    }
    
    /* Chat messages */
    .chat-message {
        margin-bottom: 24px;
        max-width: 85%;
    }
    
    .user-message {
        margin-left: auto;
        background: #f1f3f5;
    }
    
    .assistant-message {
        margin-right: auto;
        background: white;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """

def display_chat_section():
    """Display intelligent conversational chat interface with agent"""
    # Apply custom styling
    st.markdown(get_chat_style(), unsafe_allow_html=True)
    
    # Initialize session state
    if 'file_uploader_key' not in st.session_state:
        st.session_state.file_uploader_key = 0
    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = []
    if 'chat_alias_id' not in st.session_state:
        st.session_state.chat_alias_id = None
    if 'chat_context' not in st.session_state:
        st.session_state.chat_context = {}
    if 'chat_session_id' not in st.session_state:
        import uuid
        st.session_state.chat_session_id = str(uuid.uuid4())
    if 'waiting_for_user_input' not in st.session_state:
        st.session_state.waiting_for_user_input = False
    if 'detected_agent_questions' not in st.session_state:
        st.session_state.detected_agent_questions = []
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    if 'total_tokens' not in st.session_state:
        st.session_state.total_tokens = 0
    if 'chat_input_key' not in st.session_state:
        st.session_state.chat_input_key = 0
    if 'skipped_files' not in st.session_state:
        st.session_state.skipped_files = []

    # Constants
    MAX_TOKENS = 100000  # Maximum token limit
    
    # Rough token estimation factors for different file types
    TOKEN_ESTIMATION_FACTORS = {
        '.txt': 1.3,  # 1 word ≈ 1.3 tokens
        '.pdf': 1.5,  # PDF might have more special characters
        '.json': 2.0,  # JSON has more special characters and structure
        '.csv': 1.2,  # CSV is usually more compact
        '.md': 1.3,   # Similar to txt
    }

    # Chat header with New Chat button
    col1, col2 = st.columns([3, 1])
    with col1:
        st.header("💬 Chat with Agent")
    with col2:
        if st.button("🔄 New Chat", help="Start a new conversation"):
            # Save current conversation if it exists
            if st.session_state.chat_messages:
                # Ensure execution_history exists
                if 'execution_history' not in st.session_state:
                    st.session_state.execution_history = []
                
                # Only save if this conversation isn't already in history
                if (not st.session_state.execution_history or 
                    st.session_state.execution_history[-1]['session_id'] != st.session_state.chat_session_id):
                    history_entry = {
                        'timestamp_start': st.session_state.chat_messages[0]['timestamp'] if st.session_state.chat_messages else datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'timestamp_last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'session_id': st.session_state.chat_session_id,
                        'conversation': st.session_state.chat_messages.copy(),
                        'agents_used': [{
                            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'agent_name': st.session_state.selected_agent['agentName'],
                            'agent_id': st.session_state.selected_agent['agentId'],
                            'alias_name': st.session_state.chat_alias_id
                        }],
                        'files_processed': [],
                        'execution_times': []
                    }
                    st.session_state.execution_history.append(history_entry)
            
            # Clear current chat state
            st.session_state.chat_messages = []
            st.session_state.chat_context = {}
            st.session_state.uploaded_files = {}
            st.session_state.total_tokens = 0
            st.session_state.skipped_files = []
            st.session_state.file_uploader_key += 1
            # Generate new session ID
            import uuid
            st.session_state.chat_session_id = str(uuid.uuid4())
            st.session_state.waiting_for_user_input = False
            st.session_state.detected_agent_questions = []
            st.rerun()

    if not st.session_state.selected_agent:
        st.info("Please select an agent first.")
        return

    # Chat messages container
    chat_container = st.container()
    
    with chat_container:
        # Display messages
        if not st.session_state.chat_messages:
            st.markdown("""
                <div style='text-align: center; color: #6c757d; padding: 40px;'>
                    <h3>👋 Welcome!</h3>
                    <p>Start typing or upload files to begin the conversation.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            for message in st.session_state.chat_messages:
                if message.get("is_context"):
                    st.markdown(f"""
                        <div style='text-align: center; padding: 10px; color: #6c757d; font-size: 0.9em;'>
                            {message['content']}
                        </div>
                    """, unsafe_allow_html=True)
                    continue
                
                with st.chat_message(message["role"]):
                    if message["role"] == "user":
                        st.write(message["content"])
                        if message.get("has_file"):
                            st.markdown(f"""
                                <div class='file-indicator'>
                                    📎 {', '.join(message.get('file_names', []))}
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        if message.get("agent_result"):
                            display_smart_agent_result(message["agent_result"])
                        else:
                            st.write(message["content"])

    # Input container
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    
    # Input area
    placeholder = "Type your message and press Enter to send..."
    if st.session_state.waiting_for_user_input:
        placeholder = "Answer the agent's questions above..."

    # Input area and send button first
    st.subheader("Send message")

    user_input = st.chat_input(
        placeholder=placeholder,
        key=f"chat_input_{st.session_state.chat_input_key}"
    )

    # cols = st.columns([9, 1])
    
    # with cols[0]:
    #     user_input = st.text_area(
    #         "Message",
    #         key=f"chat_input_{st.session_state.chat_input_key}",
    #         placeholder=placeholder,
    #         height=100,
    #         label_visibility="collapsed"
    #     )

    # with cols[1]:
    #     send_button = st.button(
    #         "Send",
    #         key="send_button",
    #         type="primary",
    #         use_container_width=True
    #     )

    # File upload expander after input area
    with st.expander("📎 Upload files", expanded=False):
        uploaded_files = st.file_uploader(
            "Upload files",
            type=['txt', 'pdf', 'json', 'csv', 'md'],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key=f"chat_file_uploader_{st.session_state.file_uploader_key}"
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Handle file uploads
    if uploaded_files:
        st.session_state.skipped_files = []  # Reset skipped files list
        
        for uploaded_file in uploaded_files:
            if uploaded_file.name not in st.session_state.uploaded_files:
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    file_content = uploaded_file.getvalue()
                    file_processor = FileProcessor()
                    
                    # Process file based on type
                    processor_map = {
                        '.txt': file_processor.process_text_file,
                        '.pdf': file_processor.process_pdf_file,
                        '.json': file_processor.process_json_file,
                        '.csv': file_processor.process_csv_file,
                        '.md': file_processor.process_markdown_file
                    }
                    
                    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                    if file_ext in processor_map:
                        try:
                            processed_file = processor_map[file_ext](file_content, uploaded_file.name)
                            if processed_file:
                                actual_tokens = calculate_file_tokens(processed_file['file_content'], file_ext)
                                
                                if st.session_state.total_tokens + actual_tokens <= MAX_TOKENS:
                                    processed_file['token_count'] = actual_tokens
                                    st.session_state.total_tokens += actual_tokens
                                    st.session_state.uploaded_files[uploaded_file.name] = processed_file
                                    st.success(f"✅ Processed {uploaded_file.name}: {actual_tokens:,} tokens")
                                else:
                                    st.session_state.skipped_files.append(uploaded_file.name)
                                    st.warning(f"⚠️ Skipped {uploaded_file.name}: Would exceed token limit")
                        
                        except Exception as e:
                            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                            st.session_state.skipped_files.append(uploaded_file.name)

    # Process input when either Enter is pressed or Send button is clicked
    # if (user_input and user_input.strip() and (
    #     (user_input.endswith('\n') and not user_input.endswith('\n\n')) or 
    #     send_button
    # )):
    if user_input:
        cleaned_input = user_input.strip()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Prepare file data if files were uploaded
        file_data = None
        if st.session_state.uploaded_files:
            combined_content = "\n\n=== FILE SEPARATOR ===\n\n".join(
                f"File: {filename}\nContent:\n{file_info['file_content']}"
                for filename, file_info in st.session_state.uploaded_files.items()
            )
            file_data = {
                'file_type': 'multi',
                'file_name': ', '.join(st.session_state.uploaded_files.keys()),
                'file_content': combined_content,
                'metadata': {
                    'file_count': len(st.session_state.uploaded_files),
                    'total_tokens': st.session_state.total_tokens
                }
            }
        
        # Add message to chat
        user_msg = {
            "role": "user",
            "content": cleaned_input,
            "timestamp": timestamp,
            "has_file": bool(st.session_state.uploaded_files),
            "file_names": list(st.session_state.uploaded_files.keys()) if st.session_state.uploaded_files else [],
            "file_data": file_data,
            "is_follow_up": len(st.session_state.chat_messages) > 0
        }
        st.session_state.chat_messages.append(user_msg)
        # logger.info(f"User message: {user_msg}")
        
        # Execute agent
        execute_smart_chat_agent(cleaned_input, file_data)
        
        # Clear files and increment keys to force reset
        st.session_state.uploaded_files = {}
        st.session_state.total_tokens = 0
        st.session_state.chat_input_key += 1
        st.session_state.file_uploader_key += 1
        st.rerun()

def display_content_with_formatting(content_str: str):
    """Display content with intelligent formatting"""
    if not content_str or content_str.strip() == "":
        st.info("No content returned")
        return
    
    # Pre-process the content string
    # Replace \n with actual newlines
    content_str = content_str.replace('\\n', '\n')
    # Replace \t with actual tabs
    content_str = content_str.replace('\\t', '\t')
    # Handle other special characters
    content_str = content_str.replace('\\r', '\r')
    
    # Try to parse as JSON first
    try:
        content_json = json.loads(content_str)
        if isinstance(content_json, (dict, list)):
            st.json(content_json)
        else:
            st.markdown(str(content_json))
    except (json.JSONDecodeError, TypeError):
        # If not JSON, check if it looks like structured text
        if any(marker in content_str for marker in ['#', '##', '###', '*', '-', '1.', '\n-', '\n1.']):
            # Clean up markdown formatting
            # Fix numbered lists that might have escaped newlines
            content_str = re.sub(r'\\n(\d+\.)', r'\n\1', content_str)
            # Fix bullet points that might have escaped newlines
            content_str = re.sub(r'\\n-', r'\n-', content_str)
            # Ensure proper spacing for headers
            content_str = re.sub(r'\\n(#+)', r'\n\1 ', content_str)
            # Looks like markdown
            st.markdown(content_str, unsafe_allow_html=False)
        elif content_str.startswith('{') or content_str.startswith('['):
            # Might be malformed JSON, show in code block
            st.code(content_str, language='json')
        else:
            # Plain text with proper newline handling
            st.text(content_str)

def display_history_section():
    """Display execution history with enhanced filtering"""
    st.header("📈 Chat History")
    
    if not st.session_state.execution_history:
        st.info("No chat history available yet. Start a conversation to see it here.")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Filter by date
        date_range = st.selectbox("Filter by Date", ["All", "Today", "Last 7 days", "Last 30 days"])
    
    with col2:
        # Filter by agent
        all_agents = []
        for entry in st.session_state.execution_history:
            for agent in entry['agents_used']:
                all_agents.append(agent['agent_name'])
        unique_agents = ["All"] + list(set(all_agents))
        selected_agent_filter = st.selectbox("Filter by Agent", unique_agents)
    
    with col3:
        # Clear history
        if st.button("🗑️ Clear History", key="clear_history"):
            st.session_state.execution_history = []
            st.rerun()
    
    # Display history entries
    for idx, entry in enumerate(st.session_state.execution_history):
        with st.expander(f"💬 Chat Session {idx + 1} - Started: {entry['timestamp_start']}", expanded=idx == len(st.session_state.execution_history) - 1):
            # Session info
            st.markdown("#### Session Information")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Started:** {entry['timestamp_start']}")
                st.write(f"**Last Updated:** {entry['timestamp_last_update']}")
                st.write(f"**Session ID:** `{entry['session_id']}`")
            with col2:
                st.write(f"**Messages:** {len(entry['conversation'])}")
                st.write(f"**Files Processed:** {len(entry['files_processed'])}")
                avg_time = sum(entry['execution_times']) / len(entry['execution_times']) if entry['execution_times'] else 0
                st.write(f"**Average Response Time:** {avg_time:.2f}s")
            
            # Agents used
            st.markdown("#### 🤖 Agents Used")
            for agent in entry['agents_used']:
                st.markdown(f"""
                    - **{agent['agent_name']}** ({agent['alias_name']})
                    - Started using at: {agent['timestamp']}
                """)
            
            # Files processed
            if entry['files_processed']:
                st.markdown("#### 📁 Files Processed")
                for file in entry['files_processed']:
                    st.markdown(f"""
                        - **{file['file_info']}** ({file['file_type']})
                        - Processed at: {file['timestamp']}
                    """)
            
            # Display conversation
            st.markdown("#### 💬 Conversation")
            for msg in entry['conversation']:
                with st.chat_message(msg["role"]):
                    if msg["role"] == "user":
                        st.write(msg["content"])
                        if msg.get("has_file"):
                            st.caption(f"📎 Files: {', '.join(msg.get('file_names', []))}")
                        if msg.get("timestamp"):
                            st.caption(f"🕒 {msg['timestamp']}")
                    else:  # assistant
                        if msg.get("agent_result"):
                            display_smart_agent_result(msg["agent_result"])
                        else:
                            st.write(msg["content"])
                        if msg.get("timestamp"):
                            st.caption(f"🕒 {msg['timestamp']}")

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="AWS Bedrock Agents - Intelligent Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    init_session_state()
    
    # AWS Configuration section in sidebar
    is_logged_in = setup_sidebar()
    
    if not is_logged_in:
        st.info("👈 Please configure your AWS credentials in the sidebar to get started.")
        
        # Show getting started guide
        with st.expander("🚀 Getting Started Guide", expanded=True):
            st.markdown("""
            **Step 1: Configure AWS Credentials**
            - Set up your AWS credentials in the sidebar
            - Select the appropriate region where your agents are deployed
            
            **Step 2: Select Your Agent**
            - Choose from available agents in your account
            - Review the agent configuration and capabilities
            
            **Step 3: Start Conversing**
            - Upload files (TXT, PDF, JSON, CSV, MD) or enter direct input
            - Ask questions, request analysis, or seek assistance
            - Use the conversational interface for interactive sessions
            """)
        return

    # Main title
    st.title("🤖 AWS Bedrock Agents - Intelligent Assistant")
    st.markdown("Interact with your AWS Bedrock Agents for intelligent analysis and assistance.")

    # Display content based on selected page
    if st.session_state.selected_page == "🤖 Agents":
        display_agents_section()
    elif st.session_state.selected_page == "💬 Chat":
        if not st.session_state.selected_agent:
            st.warning("⚠️ Please select an agent in the Agents section first.")
        else:
            display_chat_section()
    elif st.session_state.selected_page == "📈 History":
        display_history_section()

if __name__ == "__main__":
    main()