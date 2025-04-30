export interface LibraryItem {
  id: string;
  title: string;
  category: string;
  content: string;
  timestamp: number;
  references?: string[];
  researchId?: string;
}
