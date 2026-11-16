import { Link } from "react-router-dom";

// The three flows from the user-flow diagram, one large button each.
const FLOWS = [
  { to: "/scan", title: "Scan a plant", text: "Take or upload a photo of a leaf to check for disease or pests.", primary: true },
  { to: "/plants", title: "My plants", text: "See your saved plants, their photo history and care plans." },
  { to: "/outbreaks", title: "Local outbreaks", text: "Enter your postal code to see diseases reported near you." },
];

export function HomePage() {
  return (
    <>
      <h1>What would you like to do?</h1>
      <div className="stack">
        {FLOWS.map((flow) => (
          <Link key={flow.to} to={flow.to} className={flow.primary ? "card tile tile-primary" : "card tile"}>
            <strong>{flow.title}</strong>
            <span>{flow.text}</span>
          </Link>
        ))}
      </div>
    </>
  );
}
