import React, { useState, useEffect } from 'react';
import { useNavigate, useParams, useLocation } from 'react-router-dom';
import LibraryComponent from '../components/LibraryComponent';

// Define the LibraryItem interface
interface LibraryItem {
  id: string;
  title: string;
  category: string;
  content: string;
  timestamp: number;
}

interface LibraryPageProps {
  libraryItems: LibraryItem[];
  categoryFilter: string | null;
  setCategoryFilter: (category: string | null) => void;
  setSelectedItem: (item: LibraryItem) => void;
}

const LibraryPage: React.FC<LibraryPageProps> = ({
  libraryItems,
  categoryFilter,
  setCategoryFilter,
  setSelectedItem
}) => {
  const navigate = useNavigate();
  const { category } = useParams<{ category?: string }>();
  const location = useLocation();
  const [mounted, setMounted] = useState(false);
  
  // Set mounted state to trigger re-render
  useEffect(() => {
    setMounted(true);
    return () => setMounted(false);
  }, []);
  
  // Use URL parameter if available
  useEffect(() => {
    if (category && category !== categoryFilter) {
      setCategoryFilter(category);
    } else if (!category && categoryFilter) {
      setCategoryFilter(null);
    }
  }, [category, categoryFilter, setCategoryFilter]);

  // Force a re-render when location changes
  useEffect(() => {
    // This will trigger a re-render when navigating to this page
    console.log("Library page location changed:", location.pathname);
  }, [location]);

  return (
    <div className="w-full min-h-screen">
      <LibraryComponent
        libraryItems={libraryItems}
        categoryFilter={categoryFilter}
        setCategoryFilter={setCategoryFilter}
        setSelectedItem={setSelectedItem}
        key={`library-${mounted}-${libraryItems.length}-${categoryFilter || 'all'}`}
      />
    </div>
  );
};

export default LibraryPage;
