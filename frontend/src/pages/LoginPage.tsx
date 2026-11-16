import { useState, type FormEvent } from "react";
import { Navigate, useLocation, useNavigate } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { Notice } from "../components/ui";
import { validateEmail, validatePassword } from "../lib/validation";

// Matches the account the backend creates at start-up (backend/app/store.py).
const DEMO = { email: "demo@example.com", password: "demo-password" };

export function LoginPage() {
  const { session, signIn } = useAuth();
  const navigate = useNavigate();
  const from = (useLocation().state as { from?: string } | null)?.from ?? "/";

  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showErrors, setShowErrors] = useState(false);
  const [serverError, setServerError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  if (session) return <Navigate to={from} replace />;

  const emailCheck = validateEmail(email);
  // Only new passwords get the length rule; logging in just needs something typed.
  const passwordCheck = mode === "signup" ? validatePassword(password) : password ? { ok: true as const } : { ok: false as const, message: "Enter your password." };

  async function submit(event: FormEvent) {
    event.preventDefault();
    setShowErrors(true);
    setServerError(null);
    if (!emailCheck.ok || !passwordCheck.ok) return;
    setBusy(true);
    try {
      signIn(mode === "signup" ? await api.signup(email, password) : await api.login(email, password));
      navigate(from, { replace: true });
    } catch (e) {
      setServerError(e instanceof ApiError ? e.message : "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  async function useDemo() {
    setBusy(true);
    setServerError(null);
    try {
      signIn(await api.login(DEMO.email, DEMO.password));
      navigate(from, { replace: true });
    } catch (e) {
      setServerError(e instanceof ApiError ? e.message : "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="narrow">
      <h1>{mode === "login" ? "Log in" : "Create your account"}</h1>
      <p className="muted">Check your tomato, apple and corn plants for disease.</p>
      <form onSubmit={submit} noValidate className="card form">
        {serverError && <Notice kind="error">{serverError}</Notice>}
        <label htmlFor="email">Email</label>
        <input id="email" type="email" autoComplete="email" inputMode="email" value={email}
          onChange={(e) => setEmail(e.target.value)}
          aria-invalid={showErrors && !emailCheck.ok} aria-describedby="email-error" />
        <p id="email-error" className="field-error">{showErrors && !emailCheck.ok ? emailCheck.message : ""}</p>

        <label htmlFor="password">Password</label>
        <input id="password" type="password" value={password}
          autoComplete={mode === "signup" ? "new-password" : "current-password"}
          onChange={(e) => setPassword(e.target.value)}
          aria-invalid={showErrors && !passwordCheck.ok} aria-describedby="password-error" />
        <p id="password-error" className="field-error">
          {showErrors && !passwordCheck.ok ? passwordCheck.message : mode === "signup" ? "At least 8 characters." : ""}
        </p>

        <button className="btn btn-primary" disabled={busy}>
          {busy ? "Please wait…" : mode === "login" ? "Log in" : "Sign up"}
        </button>
      </form>
      <button className="link-btn center" onClick={() => { setMode(mode === "login" ? "signup" : "login"); setShowErrors(false); setServerError(null); }}>
        {mode === "login" ? "New here? Create an account" : "Already have an account? Log in"}
      </button>
      {/* Only shown while developing (npm run dev), never in the built app. */}
      {import.meta.env.DEV && (
        <section className="card center">
          <p className="muted small">Testing? Use the demo account: {DEMO.email} / {DEMO.password}</p>
          <button type="button" className="btn btn-secondary" disabled={busy} onClick={useDemo}>Log in as demo user</button>
        </section>
      )}
    </div>
  );
}
