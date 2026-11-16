import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { Navigate, Outlet, useLocation } from "react-router-dom";
import { api, loadSession, setSessionExpiredHandler, storeSession } from "../api/client";
import type { Session } from "../api/types";

interface AuthValue {
  session: Session | null;
  signIn: (session: Session) => void;
  signOut: () => void;
}

const AuthContext = createContext<AuthValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(loadSession);

  const signIn = useCallback((next: Session) => {
    storeSession(next);
    setSession(next);
  }, []);

  const signOut = useCallback(() => {
    api.logout().catch(() => {}); // best effort; the local session is cleared either way
    storeSession(null);
    setSession(null);
  }, []);

  useEffect(() => {
    setSessionExpiredHandler(() => {
      storeSession(null);
      setSession(null);
    });
  }, []);

  const value = useMemo(() => ({ session, signIn, signOut }), [session, signIn, signOut]);
  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthValue {
  const value = useContext(AuthContext);
  if (!value) throw new Error("useAuth must be used inside AuthProvider");
  return value;
}

/** Wraps the signed-in pages. Sends everyone else to the login page and remembers where they were going. */
export function RequireAuth() {
  const { session } = useAuth();
  const location = useLocation();
  if (!session) return <Navigate to="/login" replace state={{ from: location.pathname + location.search }} />;
  return <Outlet />;
}
