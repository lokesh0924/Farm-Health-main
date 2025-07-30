import React, { createContext, useContext } from 'react';

const AuthContext = createContext({ currentUser: null });

export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
  // Always provide null for currentUser since auth is removed
  const value = { currentUser: null };
  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}; 