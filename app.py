from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import uuid
import threading
from datetime import datetime
import sqlite3
import json
import os
from werkzeug.utils import secure_filename
from draftagent import conduct_research_workflow, select_content_style
from DeepWebResearcher.DeepWebResearcher.rag import get_rag_pipeline

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Database configuration
DB_FILE = "research_database.sqlite"

# Upload folder for PDFs
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Dictionary to store research results by ID (in-memory cache)
research_results = {}

# Dictionary to store PDF metadata and paths
pdf_library = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Initialize database
def init_db():
    """Initialize the SQLite database with required tables"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create tables if they don't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS drafts (
        draft_id TEXT PRIMARY KEY,
        title TEXT NOT NULL,
        tags TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT,
        research_id TEXT,
        query TEXT,
        content_style TEXT,
        draft_content TEXT,
        reference_list TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS playlists (
        playlist_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        description TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS playlist_drafts (
        playlist_id TEXT,
        draft_id TEXT,
        added_at TEXT NOT NULL,
        PRIMARY KEY (playlist_id, draft_id),
        FOREIGN KEY (playlist_id) REFERENCES playlists(playlist_id) ON DELETE CASCADE,
        FOREIGN KEY (draft_id) REFERENCES drafts(draft_id) ON DELETE CASCADE
    )
    ''')
    
    # New table for PDF documents
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS pdf_documents (
        pdf_id TEXT PRIMARY KEY,
        filename TEXT NOT NULL,
        title TEXT,
        description TEXT,
        file_path TEXT NOT NULL,
        uploaded_at TEXT NOT NULL,
        tags TEXT,
        metadata TEXT
    )
    ''')
    
    # New table for research-PDF associations
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS research_pdfs (
        research_id TEXT,
        pdf_id TEXT,
        PRIMARY KEY (research_id, pdf_id),
        FOREIGN KEY (pdf_id) REFERENCES pdf_documents(pdf_id) ON DELETE CASCADE
    )
    ''')
    
    # Enable foreign key support
    cursor.execute("PRAGMA foreign_keys = ON")
    
    conn.commit()
    conn.close()
    
    print(f"Database initialized: {DB_FILE}")

def get_db_connection():
    """Get a database connection with foreign key support enabled"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def process_research_in_background(research_id, query, content_style, pdf_ids=None):
    """Background task to process research request with optional PDF context"""
    try:
        print("\n" + "="*50)
        print(f"PROCESSING RESEARCH: {research_id}")
        print("="*50)
        print(f"Query: {query}")
        print(f"Content Style: {content_style}")
        if pdf_ids:
            print(f"Using PDFs: {pdf_ids}")
        print("-"*50)
        
        # Update status to processing
        research_results[research_id]["status"] = "processing"
        research_results[research_id]["processing_started"] = datetime.now().isoformat()
        
        # Get PDF context if PDF IDs are provided
        pdf_context = ""
        if pdf_ids and len(pdf_ids) > 0:
            rag_pipeline = get_rag_pipeline()
            
            # Store PDF-research association in database
            conn = get_db_connection()
            cursor = conn.cursor()
            
            try:
                for pdf_id in pdf_ids:
                    # Get PDF file path from database
                    cursor.execute('SELECT file_path FROM pdf_documents WHERE pdf_id = ?', (pdf_id,))
                    pdf_row = cursor.fetchone()
                    
                    if pdf_row:
                        # Add to research_pdfs table
                        cursor.execute(
                            'INSERT OR IGNORE INTO research_pdfs (research_id, pdf_id) VALUES (?, ?)',
                            (research_id, pdf_id)
                        )
                
                conn.commit()
                
                # Get relevant context from the vector store
                pdf_context = rag_pipeline.get_relevant_context(query)
                print(f"Retrieved {len(pdf_context)} characters of PDF context")
                
            except Exception as e:
                print(f"Error processing PDF context: {str(e)}")
            finally:
                conn.close()
        
        # Conduct the research with PDF context
        result = conduct_research_workflow(query, content_style, pdf_context)
        
        # Update the research results with the complete data
        research_results[research_id].update(result)
        research_results[research_id]["status"] = "completed"
        research_results[research_id]["completed_at"] = datetime.now().isoformat()
        
        print("\n" + "="*50)
        print(f"RESEARCH COMPLETED: {research_id}")
        print("="*50)
    except Exception as e:
        print(f"ERROR in research {research_id}: {str(e)}")
        research_results[research_id]["status"] = "error"
        research_results[research_id]["error"] = str(e)
        research_results[research_id]["error_at"] = datetime.now().isoformat()

@app.route('/research/start', methods=['POST'])
def start_research():
    """
    Endpoint to initiate research based on query and content style

    Expected JSON body:
    {
        "query": "Your research query",
        "style": 1,  # 1=blog post, 2=detailed report, 3=executive summary
        "pdf_ids": ["pdf-id-1", "pdf-id-2"]  # Optional
    }
    """
    data = request.json
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    query = data.get('query')
    style_number = data.get('style', 1)
    pdf_ids = data.get('pdf_ids', [])
    
    if not query:
        return jsonify({"error": "Missing required parameter: query"}), 400
    
    try:
        style_number = int(style_number)
        if style_number not in [1, 2, 3]:
            return jsonify({"error": "Style number must be between 1 and 3"}), 400
    except ValueError:
        return jsonify({"error": "Style number must be an integer"}), 400
    
    # Validate PDF IDs if provided
    if pdf_ids:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            placeholders = ','.join(['?'] * len(pdf_ids))
            cursor.execute(f'SELECT pdf_id FROM pdf_documents WHERE pdf_id IN ({placeholders})', pdf_ids)
            valid_pdf_ids = [row['pdf_id'] for row in cursor.fetchall()]
            
            invalid_pdfs = [pdf_id for pdf_id in pdf_ids if pdf_id not in valid_pdf_ids]
            if invalid_pdfs:
                return jsonify({
                    "error": "Some PDF IDs are invalid",
                    "invalid_pdfs": invalid_pdfs
                }), 400
                
            # Use only valid PDF IDs
            pdf_ids = valid_pdf_ids
        except Exception as e:
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        finally:
            conn.close()
    
    content_style = select_content_style(style_number)
    
    # Generate a unique ID for this research
    research_id = str(uuid.uuid4())
    
    # Initialize the research results with status information
    research_results[research_id] = {
        "query": query,
        "content_style": content_style,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "optimized_query": "",
        "research_output": "",
        "claims": [],
        "verification_results": [],
        "references": [],
        "fact_check_report": "",
        "draft_content": "",
        "pdf_ids": pdf_ids
    }
    
    # Start the research in a background thread
    thread = threading.Thread(
        target=process_research_in_background,
        args=(research_id, query, content_style, pdf_ids)
    )
    thread.daemon = True
    thread.start()
    
    # Return the research ID and initial status immediately
    return jsonify({
        "status": "success",
        "message": "Research initiated successfully",
        "research_id": research_id,
        "research_status": "queued",
        "created_at": research_results[research_id]["created_at"],
        "pdf_count": len(pdf_ids) if pdf_ids else 0
    })

@app.route('/research/results/<research_id>', methods=['GET'])
def get_research_results(research_id):
    """
    Endpoint to get all research results for a given research ID
    """
    if research_id not in research_results:
        return jsonify({"error": "Research ID not found"}), 404
    
    result = research_results[research_id]
    status = result.get("status", "unknown")
    
    # If research is not completed yet
    if status in ["queued", "processing"]:
        return jsonify({
            "research_id": research_id,
            "status": status,
            "message": "Research is still in progress",
            "created_at": result.get("created_at", ""),
            "processing_started": result.get("processing_started", ""),
            "query": result.get("query", ""),
            "content_style": result.get("content_style", ""),
            "pdf_ids": result.get("pdf_ids", [])
        })
    
    # If there was an error
    if status == "error":
        return jsonify({
            "research_id": research_id,
            "status": "error",
            "message": "An error occurred during research",
            "error": result.get("error", "Unknown error"),
            "created_at": result.get("created_at", ""),
            "error_at": result.get("error_at", ""),
            "query": result.get("query", ""),
            "content_style": result.get("content_style", ""),
            "pdf_ids": result.get("pdf_ids", [])
        })
    
    # Format the response to include all components for completed research
    response = {
        "research_id": research_id,
        "status": status,
        "created_at": result.get("created_at", ""),
        "completed_at": result.get("completed_at", ""),
        "query": {
            "original": result.get("query", ""),
            "optimized": result.get("optimized_query", "")
        },
        "research_output": result.get("research_output", ""),
        "fact_check": {
            "report": result.get("fact_check_report", ""),
            "verification_results": result.get("verification_results", [])
        },
        "content": {
            "style": result.get("content_style", ""),
            "draft": result.get("draft_content", "")
        },
        "references": result.get("references", []),
        "pdf_ids": result.get("pdf_ids", [])
    }
    
    return jsonify(response)

# PDF Management APIs

@app.route('/pdfs/upload', methods=['POST'])
def upload_pdf():
    """
    Endpoint to upload a PDF file
    """
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        # Generate a unique ID for this PDF
        pdf_id = str(uuid.uuid4())
        
        # Secure the filename
        filename = secure_filename(file.filename)
        
        # Create a unique filename to avoid collisions
        unique_filename = f"{pdf_id}_{filename}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        # Save the file
        file.save(file_path)
        
        # Get metadata from form data
        title = request.form.get('title', filename)
        description = request.form.get('description', '')
        tags = request.form.get('tags', '[]')  # JSON string of tags
        
        try:
            tags_list = json.loads(tags)
            if not isinstance(tags_list, list):
                tags_list = []
        except json.JSONDecodeError:
            tags_list = []
        
        # Current timestamp
        now = datetime.now().isoformat()
        
        # Connect to database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # Insert PDF info into database
            cursor.execute('''
            INSERT INTO pdf_documents (
                pdf_id, filename, title, description, file_path, uploaded_at, tags, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                pdf_id,
                filename,
                title,
                description,
                file_path,
                now,
                json.dumps(tags_list),
                json.dumps({"original_filename": filename})
            ))
            
            conn.commit()
            
            # Process the PDF with the RAG pipeline
            try:
                rag_pipeline = get_rag_pipeline()
                chunk_count = rag_pipeline.process_pdf(file_path, {"pdf_id": pdf_id, "title": title})
                
                # Update metadata with chunk count
                cursor.execute('''
                UPDATE pdf_documents 
                SET metadata = ? 
                WHERE pdf_id = ?
                ''', (
                    json.dumps({
                        "original_filename": filename,
                        "chunk_count": chunk_count
                    }),
                    pdf_id
                ))
                
                conn.commit()
                
                return jsonify({
                    "status": "partial_success",
                    "message": f"PDF uploaded but processing failed: {str(e)}",
                    "pdf_id": pdf_id,
                    "title": title
                })
            except Exception as e: return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500
                
        except Exception as e:
            conn.rollback()
            return jsonify({"error": f"Database error: {str(e)}"}), 500
        finally:
            conn.close()
    
    return jsonify({"error": "Invalid file type. Only PDF files are allowed."}), 400

@app.route('/pdfs', methods=['GET'])
def get_all_pdfs():
    """
    Endpoint to get all PDFs in the library
    
    Query parameters:
    - tag: Filter PDFs by tag (optional)
    """
    tag_filter = request.args.get('tag')
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        if tag_filter:
            # SQLite doesn't have native JSON functions, so we'll filter in Python
            cursor.execute('SELECT * FROM pdf_documents ORDER BY uploaded_at DESC')
            rows = cursor.fetchall()
            
            # Convert rows to dictionaries and filter by tag
            pdfs_list = []
            for row in rows:
                pdf = dict(row)
                tags = json.loads(pdf['tags'] or '[]')
                pdf['tags'] = tags
                pdf['metadata'] = json.loads(pdf['metadata'] or '{}')
                
                if tag_filter in tags:
                    pdfs_list.append(pdf)
        else:
            # Get all PDFs
            cursor.execute('SELECT * FROM pdf_documents ORDER BY uploaded_at DESC')
            rows = cursor.fetchall()
            
            # Convert rows to dictionaries
            pdfs_list = []
            for row in rows:
                pdf = dict(row)
                pdf['tags'] = json.loads(pdf['tags'] or '[]')
                pdf['metadata'] = json.loads(pdf['metadata'] or '{}')
                pdfs_list.append(pdf)
        
        return jsonify({
            "count": len(pdfs_list),
            "pdfs": pdfs_list
        })
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    finally:
        conn.close()

@app.route('/pdfs/<pdf_id>', methods=['GET'])
def get_pdf_by_id(pdf_id):
    """
    Endpoint to get a specific PDF by ID
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('SELECT * FROM pdf_documents WHERE pdf_id = ?', (pdf_id,))
        row = cursor.fetchone()
        
        if not row:
            return jsonify({"error": "PDF ID not found"}), 404
        
        pdf = dict(row)
        pdf['tags'] = json.loads(pdf['tags'] or '[]')
        pdf['metadata'] = json.loads(pdf['metadata'] or '{}')
        
        return jsonify(pdf)
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    finally:
        conn.close()

@app.route('/pdfs/<pdf_id>/download', methods=['GET'])
def download_pdf(pdf_id):
    """
    Endpoint to download a PDF file
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('SELECT file_path, filename FROM pdf_documents WHERE pdf_id = ?', (pdf_id,))
        row = cursor.fetchone()
        
        if not row:
            return jsonify({"error": "PDF ID not found"}), 404
        
        file_path = row['file_path']
        original_filename = row['filename']
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({"error": "PDF file not found on server"}), 404
        
        # Get directory and filename from path
        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        
        return send_from_directory(
            directory, 
            filename, 
            as_attachment=True, 
            download_name=original_filename
        )
    except Exception as e:
        return jsonify({"error": f"Error downloading PDF: {str(e)}"}), 500
    finally:
        conn.close()

@app.route('/pdfs/<pdf_id>', methods=['DELETE'])
def delete_pdf(pdf_id):
    """
    Endpoint to delete a PDF from the library
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get file path before deleting
        cursor.execute('SELECT file_path FROM pdf_documents WHERE pdf_id = ?', (pdf_id,))
        row = cursor.fetchone()
        
        if not row:
            return jsonify({"error": "PDF ID not found"}), 404
        
        file_path = row['file_path']
        
        # Begin transaction
        conn.execute('BEGIN TRANSACTION')
        
        # Delete from database
        cursor.execute('DELETE FROM pdf_documents WHERE pdf_id = ?', (pdf_id,))
        
        # Commit transaction
        conn.commit()
        
        # Delete file from disk
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return jsonify({
            "status": "success",
            "message": "PDF deleted successfully"
        })
    except Exception as e:
        conn.rollback()
        return jsonify({"error": f"Error deleting PDF: {str(e)}"}), 500
    finally:
        conn.close()

@app.route('/pdfs/<pdf_id>', methods=['PUT'])
def update_pdf_metadata(pdf_id):
    """
    Endpoint to update PDF metadata
    
    Expected JSON body:
    {
        "title": "Updated title",
        "description": "Updated description",
        "tags": ["tag1", "tag2"]
    }
    """
    data = request.json
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check if PDF exists
        cursor.execute('SELECT pdf_id FROM pdf_documents WHERE pdf_id = ?', (pdf_id,))
        if not cursor.fetchone():
            return jsonify({"error": "PDF ID not found"}), 404
        
        # Begin transaction
        conn.execute('BEGIN TRANSACTION')
        
        # Update fields
        updates = []
        params = []
        
        if 'title' in data:
            updates.append("title = ?")
            params.append(data['title'])
        
        if 'description' in data:
            updates.append("description = ?")
            params.append(data['description'])
        
        if 'tags' in data:
            updates.append("tags = ?")
            params.append(json.dumps(data['tags']))
        
        # Add pdf_id to params
        params.append(pdf_id)
        
        # Execute update
        if updates:
            query = f"UPDATE pdf_documents SET {', '.join(updates)} WHERE pdf_id = ?"
            cursor.execute(query, params)
        
        # Commit transaction
        conn.commit()
        
        # Get updated PDF
        cursor.execute('SELECT * FROM pdf_documents WHERE pdf_id = ?', (pdf_id,))
        updated_pdf = dict(cursor.fetchone())
        updated_pdf['tags'] = json.loads(updated_pdf['tags'] or '[]')
        updated_pdf['metadata'] = json.loads(updated_pdf['metadata'] or '{}')
        
        return jsonify({
            "status": "success",
            "message": "PDF metadata updated successfully",
            "pdf": updated_pdf
        })
    except Exception as e:
        conn.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    finally:
        conn.close()

@app.route('/research/<research_id>/pdfs', methods=['GET'])
def get_research_pdfs(research_id):
    """
    Endpoint to get all PDFs associated with a research
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Check if research exists
        if research_id not in research_results:
            return jsonify({"error": "Research ID not found"}), 404
        
        # Get PDFs associated with the research
        cursor.execute('''
        SELECT p.* FROM pdf_documents p
        JOIN research_pdfs rp ON p.pdf_id = rp.pdf_id
        WHERE rp.research_id = ?
        ORDER BY p.uploaded_at DESC
        ''', (research_id,))
        
        rows = cursor.fetchall()
        
        # Convert rows to dictionaries
        pdfs_list = []
        for row in rows:
            pdf = dict(row)
            pdf['tags'] = json.loads(pdf['tags'] or '[]')
            pdf['metadata'] = json.loads(pdf['metadata'] or '{}')
            pdfs_list.append(pdf)
        
        return jsonify({
            "research_id": research_id,
            "count": len(pdfs_list),
            "pdfs": pdfs_list
        })
    except Exception as e:
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    finally:
        conn.close()

@app.route('/query-pdf', methods=['POST'])
def query_pdf_directly():
    """
    Endpoint to directly query the PDF knowledge base
    
    Expected JSON body:
    {
        "query": "Your question about the PDFs",
        "pdf_ids": ["pdf-id-1", "pdf-id-2"]  # Optional, if not provided, query all PDFs
    }
    """
    data = request.json
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    query = data.get('query')
    pdf_ids = data.get('pdf_ids', [])
    
    if not query:
        return jsonify({"error": "Missing required parameter: query"}), 400
    
    try:
        rag_pipeline = get_rag_pipeline()
        
        # If specific PDF IDs are provided, we could filter results
        # but for now we'll just query the entire vector store
        result = rag_pipeline.query(query)
        
        return jsonify({
            "status": "success",
            "query": query,
            "answer": result["answer"],
            "sources": [doc["metadata"] for doc in result["documents"]]
        })
    except Exception as e:
        return jsonify({"error": f"Error querying PDFs: {str(e)}"}), 500

# Library management APIs with SQLite

@app.route('/library/save-draft', methods=['POST'])
def save_draft_to_library():
    """
    Endpoint to save a draft to the library
    
    Expected JSON body:
    {
        "research_id": "uuid-of-research",
        "title": "Custom title for the draft",
        "tags": ["tag1", "tag2"],  # Optional
        "content": "Custom content"  # Optional
    }
    """
    data = request.json
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    
    research_id = data.get('research_id')
    title = data.get('title')
    tags = data.get('tags', [])
    custom_content = data.get('content')  # Get optional content
    
    if not research_id:
        return jsonify({"error": "Missing required parameter: research_id"}), 400
    if not title:
        return jsonify({"error": "Missing required parameter: title"}), 400
    
    # Check if research exists and is completed
    if research_id not in research_results:
        return jsonify({"error": "Research ID not found"}), 404
    
    result = research_results[research_id]
    if result.get("status") != "completed":
        return jsonify({"error": "Research is not completed yet"}), 400
    
    # Generate a unique ID for this draft
    draft_id = str(uuid.uuid4())
    
    # Current timestamp
    now = datetime.now().isoformat()
    
    # Connect to database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Use custom content if provided, otherwise use the original research content
        draft_content = custom_content if custom_content is not None else result.get("draft_content", "")
        
        # Insert draft into database
        cursor.execute('''
        INSERT INTO drafts (
            draft_id, title, tags, created_at, updated_at, research_id, 
            query, content_style, draft_content, reference_list
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            draft_id, 
            title, 
            json.dumps(tags), 
            now, 
            now,
            research_id,
            result.get("query", ""),
            result.get("content_style", ""),
            draft_content,
            json.dumps(result.get("references", []))
        ))
        
        conn.commit()
        
        return jsonify({
            "status": "success",
            "message": "Draft saved to library successfully",
            "draft_id": draft_id
        })
    except Exception as e:
        conn.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500
    finally:
        conn.close()

# Cleanup function to periodically remove old research results from memory
def cleanup_old_research():
    """Remove research results older than 24 hours from memory"""
    import time
    from datetime import timedelta
    
    while True:
        try:
            current_time = datetime.now()
            to_delete = []
            
            for research_id, data in research_results.items():
                # Check if research is completed or errored
                if data.get("status") in ["completed", "error"]:
                    # Check if it has a timestamp
                    completed_at = data.get("completed_at") or data.get("error_at")
                    if completed_at:
                        completed_time = datetime.fromisoformat(completed_at)
                        # Delete if older than 24 hours
                        if current_time - completed_time > timedelta(hours=24):
                            to_delete.append(research_id)
            
            # Delete old research results
            for research_id in to_delete:
                del research_results[research_id]
                print(f"Cleaned up old research data: {research_id}")
        
        except Exception as e:
            print(f"Error in cleanup task: {str(e)}")
        
        # Sleep for 1 hour
        time.sleep(3600)

if __name__ == '__main__':
    print("\n" + "="*50)
    print("DEEP WEB RESEARCHER API SERVER")
    print("="*50)
    
    # Initialize database
    init_db()
    
    # Start cleanup thread
    cleanup_thread = threading.Thread(target=cleanup_old_research)
    cleanup_thread.daemon = True
    cleanup_thread.start()
    
    print("Server is starting...")
    print("The research agent will process requests in the background")
    print("Database initialized and ready")
    print("PDF processing enabled with RAG pipeline")
    print("="*50)
    app.run(debug=True, host='0.0.0.0', port=5000)
