import { useState } from "react";
import Sidebar from "./Sidebar.jsx";

export default function Attendance() {
  const [attendance, setAttendance] = useState([
    { subject: "Mathematics", attended: 18, total: 20 },
    { subject: "Physics", attended: 15, total: 20 },
    { subject: "Chemistry", attended: 12, total: 20 },
    { subject: "Computer Science", attended: 19, total: 20 },
    { subject: "English", attended: 14, total: 20 },
  ]);

  return (
    <div className="flex min-h-screen">
      {/* Sidebar */}
      <Sidebar />

      {/* Main content */}
      <div className="flex-1 px-8 py-6">
        <h1 className="text-2xl font-semibold mb-6">My Attendance (Subject-wise)</h1>

        {/* Attendance Summary Table */}
        <div className="bg-[#F5F4E9] p-6 rounded-2xl">
          <table className="w-full border border-black text-left">
            <thead>
              <tr className="border bg-gray-100">
                <th className="border p-2">Subject</th>
                <th className="border p-2">Attended</th>
                <th className="border p-2">Total Classes</th>
                <th className="border p-2">Percentage</th>
              </tr>
            </thead>
            <tbody>
              {attendance.map((record, index) => {
                const percentage = Math.round((record.attended / record.total) * 100);
                return (
                  <tr key={index} className="border">
                    <td className="border p-2">{record.subject}</td>
                    <td className="border p-2">{record.attended}</td>
                    <td className="border p-2">{record.total}</td>
                    <td
                      className={`border p-2 font-semibold ${
                        percentage >= 75 ? "text-green-600" : "text-red-600"
                      }`}
                    >
                      {percentage}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
