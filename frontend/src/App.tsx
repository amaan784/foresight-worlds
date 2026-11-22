import Capture from "./Capture";
import Results from "./Results";
import Stage from "./Stage";

export default function App() {
  const path = location.pathname;
  if (path.startsWith("/capture")) return <Capture />;
  if (path.startsWith("/results")) return <Results />;
  return <Stage />;
}
