import React from 'react';
import ReactDOM from 'react-dom/client';
import { ConversationProvider } from '@/state/conversationStore';
import App from './App';
import './global.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ConversationProvider>
      <App />
    </ConversationProvider>
  </React.StrictMode>,
);
