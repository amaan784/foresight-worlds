import Capture from "./Capture";
import Stage from "./Stage";

export default function App() {
  return location.pathname.startsWith("/capture") ? <Capture /> : <Stage />;
}
