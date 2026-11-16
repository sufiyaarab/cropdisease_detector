import { NavLink, Outlet } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";

export function Layout() {
  const { session, signOut } = useAuth();
  return (
    <div className="app">
      <header className="topbar">
        <NavLink to="/" className="brand">Crop Disease Monitor</NavLink>
        {session && (
          <button className="link-btn" onClick={signOut} title={session.email}>Sign out</button>
        )}
      </header>
      <main className="content">
        <Outlet />
      </main>
      {session && (
        <nav className="tabbar" aria-label="Main">
          <NavLink to="/" end>Home</NavLink>
          <NavLink to="/scan">Scan</NavLink>
          <NavLink to="/plants">My plants</NavLink>
          <NavLink to="/outbreaks">Outbreaks</NavLink>
        </nav>
      )}
    </div>
  );
}
