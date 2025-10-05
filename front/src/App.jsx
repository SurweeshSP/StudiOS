import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import Landing from './Components/Landing.jsx'
import Dashboardd from './Components/Dashboardd.jsx'
import ChatPage from './Components/ChatPage.jsx'
import Suggestion from './Components/Suggestion.jsx'
import Attendance from './Components/Attendance.jsx'
import Details from './Components/Details.jsx'
import Analytics from './Components/Analytics.jsx'
import Sidebar from './Components/Sidebar.jsx'
import Navbar from "./Components/Navbar.jsx";

function App() {

  return (
    <>
    <Navbar />
    <div className="flex">
      <div className="flex-1">
        <Routes>
          <Route path="/" element={<Landing />} />
          <Route path="/Landing" element={<Landing />} />
          <Route path="/Dashboard" element={<Dashboardd />} />
          <Route path="/ChatPage" element={<ChatPage />} />
          <Route path="/Suggestion" element={<Suggestion />} />
          <Route path="/Attendance" element={<Attendance />} />
          <Route path="/Details" element={<Details />} />
          <Route path="/Analytics" element={<Analytics />} />
        </Routes>
      </div>
    </div>
    </>
  )
}

export default App
