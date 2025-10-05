import { Link } from "react-router-dom";

function Navbar() {
  return (
    <nav className="fixed top-0 left-0 w-full bg-white/80 backdrop-blur-md shadow-sm z-50 mb-30">
      <div className="flex justify-between items-center px-10 py-5">
        
        {/* Logo */}
        <Link to="/" className="text-3xl font-bold text-indigo-600 tracking-wide cursor-pointer">
          EduLoop
        </Link>

        {/* Nav Links */}
        <ul className="navcomp flex space-x-10 text-lg font-medium text-gray-500">
          <li>
            <Link
              to="/"
              className="relative cursor-pointer hover:text-indigo-600 transition duration-300"
            >
              Home
            </Link>
          </li>
          <li>
            <Link
              to="/Dashboard"
              className="relative cursor-pointer hover:text-indigo-600 transition duration-300"
            >
              Dashboard
            </Link>
          </li>
          <li>
            <Link
              to="/ChatPage"
              className="relative cursor-pointer hover:text-indigo-600 transition duration-300"
            >
              Chat
            </Link>
          </li>
          <li>
            <Link
              to="/Suggestion"
              className="relative cursor-pointer hover:text-indigo-600 transition duration-300"
            >
              Suggestion
            </Link>
          </li>
        </ul>
      </div>
    </nav>
  );
}

export default Navbar;
