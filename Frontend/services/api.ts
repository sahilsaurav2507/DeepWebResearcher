// API service for communicating with the DeepWebResearcher backend

const API_BASE_URL = 'http://127.0.0.1:5000'; // Updated to match the port in app.py

export interface ResearchResponse {
  research_id: string;
  status: string;
  message: string;
  research_status: string;
  created_at: string;
}

export interface ResearchResult {
  research_id: string;
  status: string;
  created_at: string;
  completed_at?: string;
  query: {
    original: string;
    optimized: string;
  };
  research_output: string;
  fact_check: {
    report: string;
    verification_results: any[];
  };
  content: {
    style: string;
    draft: string;
  };
  references: string[];
}

export interface SavedDraft {
  draft_id: string;
  title: string;
  tags: string[];
  created_at: string;
  updated_at: string;
  research_id: string;
  query: string;
  content_style: string;
  draft_content: string;
  references: string[];
}

// Start a new research
export const startResearch = async (query: string, style: number): Promise<ResearchResponse> => {
  try {
    const response = await fetch(`${API_BASE_URL}/research/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ query, style }),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to start research');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error starting research:', error);
    throw error;
  }
};

// Get research results
export const getResearchResults = async (researchId: string): Promise<ResearchResult> => {
  try {
    const response = await fetch(`${API_BASE_URL}/research/results/${researchId}`);
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to get research results');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error getting research results:', error);
    throw error;
  }
};

// Save draft to library
export const saveDraftToLibrary = async (
  researchId: string, 
  title: string, 
  tags: string[] = [],
  content?: string
): Promise<{ draft_id: string }> => {
  try {
    console.log("API: Saving draft with research ID:", researchId);
    console.log("API: Content length:", content?.length);
    
    const requestBody: any = { 
      research_id: researchId, 
      title, 
      tags
    };
    
    // Only include content if it's provided
    if (content !== undefined) {
      requestBody.content = content;
    }
    
    const response = await fetch(`${API_BASE_URL}/library/save-draft`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to save draft');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error saving draft:', error);
    throw error;
  }
};

// Save a copy of an existing draft with edited content
export const saveDraftCopy = async (
  title: string,
  content: string,
  contentStyle: string,
  tags: string[] = [],
  references: string[] = []
): Promise<{ draft_id: string }> => {
  try {
    const response = await fetch(`${API_BASE_URL}/library/save-copy`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        title,
        content,
        content_style: contentStyle,
        tags,
        references
      }),
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to save draft copy');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error saving draft copy:', error);
    throw error;
  }
};

// Get all drafts
export const getAllDrafts = async (): Promise<{ drafts: SavedDraft[] }> => {
  try {
    const response = await fetch(`${API_BASE_URL}/library/drafts`);
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to get drafts');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error getting drafts:', error);
    throw error;
  }
};

// Get draft by ID
export const getDraftById = async (draftId: string): Promise<SavedDraft> => {
  try {
    const response = await fetch(`${API_BASE_URL}/library/drafts/${draftId}`);
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Failed to get draft');
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error getting draft:', error);
    throw error;
  }
};
