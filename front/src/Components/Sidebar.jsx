import {Mail, User, BarChart3, Home, Calendar, Settings, Lock } from "lucide-react";
import { Link } from "react-router-dom";

function Sidebar() {
    return (
        <div className="w-20 flex flex-col items-center py-6 justify-between h-[165px]">
            <div className="flex flex-col items-center space-y-8">
                <Link to="/Landing"><Home className="w-6 h-6 cursor-pointer" /></Link>
                <Link to="/Attendance"><Calendar className="w-6 h-6 cursor-pointer" /></Link>
                <Link to="/Details"><User className="w-6 h-6 cursor-pointer" /></Link>
                <Link to="/Analytics"><BarChart3 className="w-6 h-6 cursor-pointer" /></Link>
            </div>
            <Settings className="w-6 h-6 cursor-pointer" />
        </div>
    )
}

export default Sidebar