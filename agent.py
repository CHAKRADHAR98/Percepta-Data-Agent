"""
Bimanual Robot Learning Data Agent
Main agent with Chat Protocol for ASI:One compatibility
"""
import os
import json
from datetime import datetime, timezone
from uuid import uuid4
from pathlib import Path
from dotenv import load_dotenv

from uagents import Agent, Context, Protocol
from uagents_core.contrib.protocols.chat import (
    ChatAcknowledgement,
    ChatMessage,
    EndSessionContent,
    StartSessionContent,
    TextContent,
    chat_protocol_spec,
)

from config import Config
from utils import LLM, normalize_task_name
from metta_rag import BimanualDatasetRAG
from pinata_handler import PinataHandler
from pipeline import BimanualPipelineRunner

# Load environment variables
load_dotenv()

# Initialize agent with mailbox
agent = Agent(
    name=Config.AGENT_NAME,
    seed=Config.AGENT_SEED,
    port=Config.AGENT_PORT,
    mailbox=True,
)

print(f"🤖 Agent Address: {agent.address}")

# Initialize global components
print("🔧 Initializing components...")
llm = LLM(api_key=os.getenv("ASI_ONE_API_KEY"))
rag = BimanualDatasetRAG()

# Load existing knowledge graph if available
kg_path = Config.METADATA_DIR / "knowledge_graph.json"
if kg_path.exists():
    rag.import_knowledge_graph(kg_path)

pinata = PinataHandler(
    api_key=os.getenv("PINATA_API_KEY"),
    secret_key=os.getenv("PINATA_SECRET_KEY")
)
pipeline_runner = BimanualPipelineRunner(Config)


def create_text_chat(text: str, end_session: bool = False) -> ChatMessage:
    """Create a text chat message"""
    content = [TextContent(type="text", text=text)]
    if end_session:
        content.append(EndSessionContent(type="end-session"))
    return ChatMessage(
        timestamp=datetime.now(timezone.utc),
        msg_id=uuid4(),
        content=content,
    )


# Setup chat protocol
chat_proto = Protocol(spec=chat_protocol_spec)


@chat_proto.on_message(ChatMessage)
async def handle_message(ctx: Context, sender: str, msg: ChatMessage):
    """Handle incoming chat messages - main intelligence here"""
    
    # Store session
    ctx.storage.set(str(ctx.session), sender)
    
    # Send acknowledgement
    await ctx.send(
        sender,
        ChatAcknowledgement(
            timestamp=datetime.now(timezone.utc),
            acknowledged_msg_id=msg.msg_id
        ),
    )
    
    # Process message content
    for item in msg.content:
        if isinstance(item, StartSessionContent):
            ctx.logger.info(f"Session started with {sender}")
            welcome_msg = (
                "👋 **Welcome to the Bimanual Robot Learning Data Agent!**\n\n"
                "I help you manage robot training datasets. Here's what I can do:\n\n"
                "📥 **RETRIEVE** - Find existing datasets\n"
                "   Example: *\"I need data for opening bottles\"*\n\n"
                "📤 **PROCESS** - Process new videos\n"
                "   Example: *\"Process this video: ipfs://Qm... (task: opening bottle)\"*\n\n"
                "💡 **HELP** - Learn more about me\n"
                "   Try: `/help`, `/tasks`, `/stats`\n\n"
                "What would you like to do?"
            )
            await ctx.send(sender, create_text_chat(welcome_msg))
            continue
        
        elif isinstance(item, TextContent):
            user_query = item.text.strip()
            ctx.logger.info(f"Query from {sender}: {user_query}")
            
            try:
                # Check for special commands first
                if user_query.startswith('/'):
                    response = handle_special_commands(ctx, user_query)
                    await ctx.send(sender, create_text_chat(response))
                    continue
                
                # Pattern-based intent detection (reliable fallback)
                user_query_lower = user_query.lower()
                
                # Check for PROCESS intent (highest priority)
                if ('process' in user_query_lower and 
                    ('ipfs://' in user_query or 'ipfs.io' in user_query or 
                     'pinata.cloud' in user_query or 'gateway' in user_query)):
                    intent = 'process'
                    # Extract video URL and task
                    import re
                    url_match = re.search(r'https?://[^\s]+|ipfs://[^\s]+', user_query)
                    task_match = re.search(r'\(task:\s*([^)]+)\)', user_query, re.IGNORECASE)
                    
                    intent_data = {
                        'intent': 'process',
                        'video_url': url_match.group(0) if url_match else None,
                        'task': task_match.group(1).strip() if task_match else None
                    }
                
                # Check for RETRIEVE intent
                elif any(phrase in user_query_lower for phrase in [
                    'need data', 'find data', 'search for', 'looking for',
                    'show me', 'get data', 'dataset', 'do you have'
                ]):
                    intent = 'retrieve'
                    intent_data = {'intent': 'retrieve', 'query': user_query}
                
                # Check for HELP intent
                elif any(phrase in user_query_lower for phrase in [
                    'help', 'what can you do', 'how do', 'explain',
                    'capabilities', 'instructions'
                ]):
                    intent = 'help'
                    intent_data = {'intent': 'help'}
                
                else:
                    # Try LLM classification as fallback
                    try:
                        intent_data = llm.classify_intent(user_query)
                        intent = intent_data.get('intent', 'retrieve')
                    except Exception as e:
                        ctx.logger.warning(f"LLM classification failed: {e}, defaulting to retrieve")
                        intent = 'retrieve'
                        intent_data = {'intent': 'retrieve', 'query': user_query}
                
                ctx.logger.info(f"Classified intent: {intent}")
                
                if intent == 'help':
                    # HELP MODE: Explain capabilities
                    response = handle_help_mode(ctx, user_query)
                    await ctx.send(sender, create_text_chat(response))
                
                elif intent == 'retrieve':
                    # RETRIEVE MODE: Find existing datasets
                    response = await handle_retrieve_mode(
                        ctx, user_query, intent_data, sender
                    )
                    await ctx.send(sender, create_text_chat(response))
                
                elif intent == 'process':
                    # PROCESS MODE: Process new video
                    response = await handle_process_mode(
                        ctx, user_query, intent_data, sender
                    )
                    await ctx.send(sender, create_text_chat(response))
                
                else:
                    await ctx.send(
                        sender,
                        create_text_chat(
                            "I'm not sure what you want. Please specify:\n"
                            "- To find data: \"I need data for [task]\"\n"
                            "- To process video: \"Process ipfs://... (task: [task name])\"\n"
                            "- For help: Type `/help`"
                        )
                    )
            
            except Exception as e:
                ctx.logger.error(f"Error processing query: {e}")
                await ctx.send(
                    sender,
                    create_text_chat(
                        f"❌ Sorry, I encountered an error: {str(e)}\n"
                        "Please try again or rephrase your request."
                    )
                )
        
        elif isinstance(item, EndSessionContent):
            ctx.logger.info(f"Session ended with {sender}")
        
        else:
            ctx.logger.info(f"Unexpected content from {sender}")


def handle_special_commands(ctx: Context, command: str) -> str:
    """Handle special slash commands"""
    command_lower = command.lower().strip()
    
    if command_lower == '/help':
        return (
            "🤖 **Bimanual Data Agent - Help**\n\n"
            "**What I Do:**\n"
            "I manage robot training datasets for bimanual (two-handed) manipulation tasks.\n\n"
            "**Available Commands:**\n"
            "• `/help` - Show this help message\n"
            "• `/tasks` - List all available task categories\n"
            "• `/stats` - Show knowledge graph statistics\n"
            "• `/recent` - Show recently added datasets\n\n"
            "**How to Retrieve Datasets:**\n"
            "Just tell me what task you need data for!\n"
            "Example: *\"I need data for opening bottles\"*\n\n"
            "**How to Process Videos:**\n"
            "Provide an IPFS URL and task annotation.\n"
            "Example: *\"Process ipfs://Qm... (task: opening bottle)\"*\n\n"
            "Need more help? Ask me anything!"
        )
    
    elif command_lower == '/tasks':
        response = "📋 **Available Task Categories:**\n\n"
        for category, tasks in Config.TASK_TAXONOMY.items():
            response += f"**{category}**\n"
            task_list = ", ".join(tasks[:5])  # Show first 5
            response += f"   {task_list}"
            if len(tasks) > 5:
                response += f" ... (+{len(tasks)-5} more)"
            response += "\n\n"
        response += "Ask me for any of these tasks!"
        return response
    
    elif command_lower == '/stats':
        stats = rag.get_statistics()
        response = (
            "📊 **Knowledge Graph Statistics:**\n\n"
            f"• **Total Tasks:** {stats['total_tasks']}\n"
            f"• **Total Datasets:** {stats['total_datasets']}\n\n"
        )
        if stats['tasks_by_category']:
            response += "**Datasets by Category:**\n"
            for category, count in stats['tasks_by_category'].items():
                response += f"   • {category.replace('_', ' ').title()}: {count} tasks\n"
        else:
            response += "*Upload more videos!*\n"
        return response
    
    elif command_lower == '/recent':
        all_tasks = rag.get_all_tasks()
        if not all_tasks:
            return "📭 No datasets found yet. Upload some videos to process!"
        
        response = "🕒 **Recently Available Tasks:**\n\n"
        for task in all_tasks[:10]:  # Show last 10
            datasets = rag.query_datasets_by_task(task)
            response += f"• **{task.replace('_', ' ').title()}** ({len(datasets)} dataset{'s' if len(datasets) != 1 else ''})\n"
        return response
    
    else:
        return f"❌ Unknown command: `{command}`\n\nType `/help` to see available commands."


def handle_help_mode(ctx: Context, user_query: str) -> str:
    """
    HELP MODE: Answer questions about the agent
    """
    query_lower = user_query.lower()
    
    # Check what user is asking about
    if any(word in query_lower for word in ['what', 'who', 'can you', 'do you', 'help']):
        return (
            "👋 Hi! I'm the **Bimanual Robot Learning Data Agent**.\n\n"
            "I help robotics researchers by:\n\n"
            "1️⃣ **Finding robot demonstration datasets** - Just tell me what task you need!\n"
            "2️⃣ **Processing new demonstration videos** - Upload to IPFS, I'll extract the training data\n"
            "3️⃣ **Quality validation** - I score every dataset so you know it's good\n"
            "4️⃣ **Semantic search** - Find similar tasks even if you don't know the exact name\n\n"
            "**Try these examples:**\n"
            "• *\"I need data for opening bottles\"*\n"
            "• *\"Show me warehouse picking datasets\"*\n"
            "• *\"Process ipfs://Qm... (task: pouring liquid)\"*\n"
            "• `/tasks` - See all available categories\n"
            "• `/stats` - View knowledge graph stats\n\n"
            "What would you like to do?"
        )
    
    # Default help
    return (
        "I help manage robot training datasets. Try:\n"
        "• Ask for datasets: *\"I need data for [task]\"*\n"
        "• Process video: *\"Process ipfs://... (task: [task])\"*\n"
        "• Get help: `/help`"
    )


async def handle_retrieve_mode(ctx: Context, user_query: str, 
                               intent_data: dict, sender: str) -> str:
    """
    RETRIEVE MODE: Find and return existing datasets
    """
    task = intent_data.get('task')
    
    if not task:
        # Try to extract task from query
        task = llm.extract_task_from_annotation(user_query)
    
    if not task or task == "unknown_task":
        # Show available tasks
        all_tasks = rag.get_all_tasks()
        if all_tasks:
            tasks_list = "\n".join([f"• {t.replace('_', ' ').title()}" for t in all_tasks[:10]])
            return (
                f"🔍 I couldn't identify the task. Here are available tasks:\n\n"
                f"{tasks_list}\n\n"
                f"Please specify one of these tasks."
            )
        else:
            return (
                "📭 **No datasets found in the knowledge graph yet.**\n\n"
                "To get started:\n"
                "1. Upload a demonstration video to IPFS\n"
                "2. Send me: *\"Process ipfs://[your-hash] (task: [task name])\"*\n"
                "3. I'll process it and add it to the network!\n\n"
                "Example: *\"Process ipfs://QmX... (task: opening bottle)\"*"
            )
    
    # Normalize task name
    task = normalize_task_name(task)
    
    # Query datasets
    datasets = rag.query_datasets_by_task(task)
    
    if not datasets:
        # Try to find similar tasks
        similar = rag.find_similar_tasks(task)
        
        if similar:
            similar_list = "\n".join([f"• {t.replace('_', ' ').title()}" for t in similar])
            return (
                f"❌ No datasets found for **{task.replace('_', ' ').title()}**\n\n"
                f"**Similar tasks with data:**\n{similar_list}\n\n"
                f"Would you like data for any of these?"
            )
        else:
            return (
                f"❌ No datasets found for **{task.replace('_', ' ').title()}**\n"
                f"No similar tasks found either.\n\n"
                f"💡 **Want to contribute?**\n"
                f"Upload a video: *\"Process ipfs://... (task: {task.replace('_', ' ')})\"*"
            )
    
    # Format response with datasets
    response = f"✅ Found **{len(datasets)}** dataset(s) for **{task.replace('_', ' ').title()}**:\n\n"
    
    for i, ipfs_hash in enumerate(datasets, 1):
        metadata = rag.get_dataset_metadata(ipfs_hash)
        
        response += f"📦 **Dataset {i}**\n"
        response += f"🔗 **Folder Hash:** `{ipfs_hash}`\n"
        response += f"🌐 **Browse Dataset:** https://ipfs.io/ipfs/{ipfs_hash}\n"
        
        if metadata.get('quality_score'):
            response += f"⭐ **Quality:** {metadata['quality_score']:.0f}/100\n"
        if metadata.get('duration'):
            response += f"⏱️ **Duration:** {metadata['duration']:.1f}s\n"
        
        response += f"\n📄 **Access Files:**\n"
        response += f"   • Training Data: `https://ipfs.io/ipfs/{ipfs_hash}/actions.npy`\n"
        response += f"   • Metadata: `https://ipfs.io/ipfs/{ipfs_hash}/actions.json`\n"
        response += f"   • Quality Report: `https://ipfs.io/ipfs/{ipfs_hash}/quality_report.json`\n"
        
        response += "\n"
    
    return response


async def handle_process_mode(ctx: Context, user_query: str,
                              intent_data: dict, sender: str) -> str:
    """
    PROCESS MODE: Download video, process it, upload results
    """
    video_url = intent_data.get('video_url')
    
    if not video_url:
        return (
            "❌ Please provide an IPFS video URL.\n"
            "Format: `Process this video: ipfs://Qm... (task: opening bottle)`"
        )
    
    # Extract task annotation
    task = intent_data.get('task')
    if not task:
        task = llm.extract_task_from_annotation(user_query)
    
    if not task or task == "unknown_task":
        return (
            "❌ Please specify the task.\n"
            "Example: `Process ipfs://Qm... (task: opening bottle)`"
        )
    
    task = normalize_task_name(task)
    
    # Notify user processing started
    await ctx.send(
        sender,
        create_text_chat(
            f"🚀 **Processing Started**\n\n"
            f"Task: **{task.replace('_', ' ').title()}**\n"
            f"Video: `{video_url[:50]}...`\n\n"
            f"This will take 5-10 minutes. I'll notify you when done! ⏳"
        )
    )
    
    try:
        # Step 1: Download video from IPFS
        ctx.logger.info("Downloading video from IPFS...")
        video_path = pinata.download_video(video_url, Config.VIDEOS_FROM_IPFS_DIR)
        
        if not video_path:
            return "❌ Failed to download video from IPFS. Please check the URL."
        
        # Step 2: Run pipeline
        ctx.logger.info("Running bimanual pipeline...")
        
        # Create task annotations
        category = rag.get_category_for_task(task) or "Other"
        task_annotations = {
            'task': task.replace('_', ' ').title(),
            'category': category.replace('_', ' ').title(),
            'task_id': task,
            'timestamp': datetime.now().isoformat()
        }
        
        # Process video
        result = pipeline_runner.run(video_path, task_annotations)
        
        # Step 3: Upload to IPFS
        ctx.logger.info("Uploading results to IPFS...")
        base_name = video_path.stem
        
        try:
            manifest = pinata.upload_dataset(task, base_name, Config.DATASET_DIR)
        except Exception as e:
            ctx.logger.error(f"Upload failed: {e}")
            return f"❌ Processing succeeded but upload to IPFS failed: {str(e)}\n\nThe processed files are saved locally at:\n`{Config.DATASET_DIR}`"
        
        if not manifest:
            return "❌ Processing succeeded but upload to IPFS failed.\n\nThe processed files are saved locally at:\n`{Config.DATASET_DIR}`"
        
        # Step 4: Update knowledge graph
        ctx.logger.info("Updating knowledge graph...")
        
        metadata = {
            'quality_score': result['quality_report']['quality_score'],
            'duration': result['metadata']['tracking_metrics']['total_frames'] / 
                       result['metadata']['tracking_metrics']['fps'],
            'frame_count': result['metadata']['tracking_metrics']['total_frames'],
            'timestamp': datetime.now().isoformat()
        }
        
        # Add to MeTTa knowledge graph - use folder hash
        folder_hash = manifest.get('folder_hash', 'unknown')
        rag.add_dataset(
            task=task,
            ipfs_hash=folder_hash,
            metadata=metadata
        )
        
        # Export knowledge graph
        kg_export_path = Config.METADATA_DIR / "knowledge_graph.json"
        rag.export_knowledge_graph(kg_export_path)
        
        # Step 5: Format success response
        quality_score = result['quality_report']['quality_score']
        
        if quality_score >= 80:
            quality_emoji = "🌟"
            quality_text = "Excellent"
        elif quality_score >= 60:
            quality_emoji = "✅"
            quality_text = "Good"
        else:
            quality_emoji = "⚠️"
            quality_text = "Fair"
        
        response = (
            f"✅ **Processing Complete!**\n\n"
            f"📊 **Quality Score:** {quality_emoji} {quality_score:.0f}/100 ({quality_text})\n"
            f"⏱️ **Duration:** {metadata['duration']:.1f}s\n"
            f"🎞️ **Frames:** {metadata['frame_count']}\n\n"
            f"📦 **Your Dataset on IPFS:**\n\n"
            f"🔗 **Folder Hash:** `{folder_hash}`\n"
            f"🌐 **Browse Dataset:** https://ipfs.io/ipfs/{folder_hash}\n\n"
            f"📄 **Individual Files:**\n"
        )
        
        # List available files with ipfs.io gateway
        files = manifest.get('files', {})
        if 'actions.npy' in files:
            response += f"   • **Training Data:** `https://ipfs.io/ipfs/{folder_hash}/actions.npy`\n"
        if 'actions.json' in files:
            response += f"   • **Metadata:** `https://ipfs.io/ipfs/{folder_hash}/actions.json`\n"
        if 'quality_report.json' in files:
            response += f"   • **Quality Report:** `https://ipfs.io/ipfs/{folder_hash}/quality_report.json`\n"
        if 'video_annotated.mp4' in files:
            response += f"   • **Annotated Video:** `https://ipfs.io/ipfs/{folder_hash}/video_annotated.mp4`\n"
        
        response += (
            f"\n✨ **Dataset is now discoverable in the network!**\n"
            f"Others can find it by querying for: **{task.replace('_', ' ').title()}**"
        )
        
        return response
    
    except Exception as e:
        ctx.logger.error(f"Processing error: {e}")
        return f"❌ Processing failed: {str(e)}"


@chat_proto.on_message(ChatAcknowledgement)
async def handle_ack(ctx: Context, sender: str, msg: ChatAcknowledgement):
    """Handle acknowledgements"""
    ctx.logger.info(f"Acknowledgement from {sender} for {msg.acknowledged_msg_id}")


# Register protocol
agent.include(chat_proto, publish_manifest=True)

print("✅ Agent initialized successfully!")
print(f"📍 Agent will be available at: http://localhost:{Config.AGENT_PORT}")
print(f"🔗 Connect via Agentverse Mailbox after starting")

if __name__ == "__main__":
    print("\n🚀 Starting Bimanual Data Agent...")
    print("="*70)
    agent.run()