import { Mail } from "lucide-react";
import Sidebar from "./Sidebar.jsx";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

const data = [
  { month: "Oct", users: 20 },
  { month: "Mar", users: 40 },
  { month: "Jul", users: 35 },
  { month: "Aug", users: 50 },
];

const tdata = [
  { subcode: "a", subname: "subject 1", Assignment: 3, Status: "Done" },
  { subcode: "b", subname: "subject 2", Assignment: 2, Status: "Yet to" },
  { subcode: "c", subname: "subject 3", Assignment: 3, Status: "Yet to" },
  { subcode: "d", subname: "subject 4", Assignment: 2, Status: "Done" },
];

const student = { name: "Vasantha kumar p" };

export default function Dashboardd() {
  return (
    <div className="flex min-h-screen bg-gray-50">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Content */}
      <div className="flex-1 px-8 py-6">
        {/* Header */}
        <div className="flex justify-between items-center">
          <h1 className="text-3xl font-semibold">Dashboard</h1>
          <div className="flex items-center space-x-4">
            <Mail className="w-6 h-6 text-gray-600 cursor-pointer hover:text-indigo-600 transition" />
          </div>
        </div>

        {/* Hero Section */}
        <div className="mt-6 flex flex-col lg:flex-row justify-between gap-6">
          {/* Welcome Card */}
          <div className="bg-gradient-to-r from-indigo-200 to-indigo-100 rounded-2xl p-6 w-full lg:w-2/3 relative shadow-md">
            <h2 className="text-4xl font-bold text-gray-800">
              Welcome, <br /> {student.name}
            </h2>
            <button className="mt-6 px-6 py-2 bg-indigo-600 text-white rounded-xl font-medium hover:bg-indigo-700 transition">
              Start Now
            </button>
            <img
              src="./src/assets/model.png"
              alt="Model"
              className="absolute bottom-0 right-6 h-48 w-48 object-contain"
            />
          </div>

          {/* Right Side Cards */}
          <div className="w-full lg:w-1/3 space-y-5">
            {/* Chart Card */}
            <div className="bg-white p-4 rounded-2xl shadow hover:shadow-lg transition">
              <h3 className="text-xl font-bold pb-2">Social Media Analysis</h3>
              <ResponsiveContainer width="100%" height={150}>
                <LineChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#eee" />
                  <XAxis dataKey="month" stroke="#888" />
                  <YAxis stroke="#888" />
                  <Tooltip />
                  <Line
                    type="monotone"
                    dataKey="users"
                    stroke="#6366f1"
                    strokeWidth={2}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Marks Card */}
            <div className="bg-white p-4 rounded-2xl shadow">
              <h3 className="text-xl font-bold">Marks</h3>
              <div className="mt-2">
                <p className="text-gray-600">Subject 1</p>
                <div className="w-full bg-gray-200 h-3 rounded-full mb-2">
                  <div className="bg-indigo-500 h-3 rounded-full" style={{ width: "68%" }}></div>
                </div>
                <p className="text-gray-600">Subject 2</p>
                <div className="w-full bg-gray-200 h-3 rounded-full">
                  <div className="bg-indigo-500 h-3 rounded-full" style={{ width: "70%" }}></div>
                </div>
                <span className="text-xs text-gray-400 block mt-2">Student Analysis</span>
              </div>
            </div>
          </div>
        </div>

        {/* Assignments Table */}
        <div className="bg-white w-full lg:w-2/3 mt-6 p-6 rounded-2xl shadow">
          <h3 className="font-bold text-xl mb-4">Ongoing Assignments</h3>
          <table className="w-full border-collapse">
            <thead>
              <tr className="bg-gray-100 text-left">
                <th className="py-2 px-3">Subject Code</th>
                <th className="py-2 px-3">Subject Name</th>
                <th className="py-2 px-3">Assignments</th>
                <th className="py-2 px-3">Status</th>
              </tr>
            </thead>
            <tbody>
              {tdata.map((item, index) => (
                <tr key={index} className="border-t">
                  <td className="py-2 px-3">{item.subcode}</td>
                  <td className="py-2 px-3">{item.subname}</td>
                  <td className="py-2 px-3">{item.Assignment}</td>
                  <td className="py-2 px-3">
                    <span
                      className={`px-3 py-1 rounded-full text-sm font-medium ${
                        item.Status === "Done"
                          ? "bg-green-100 text-green-600"
                          : "bg-yellow-100 text-yellow-600"
                      }`}
                    >
                      {item.Status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
