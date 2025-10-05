import Sidebar from './Sidebar.jsx'

const detailss = {
    name: "vasantha kumar p",
    rollno: "123",
    dob: "12-01-2005",
    gender: "male",
    nation: "Indian",
    contact: "9876543210",
    emcont: "9876543210",

    program: "B.Tech",
    acahist: "12th",
    enrostat: "Active",
    acaadvi: "someone",
    gragpa: "9.0",
    acastan: "good",

    studcred: "wifi",
    library: "open",
    hostel: "yes",
    hostno: "123",

    healthrec: "good",
    disable: "No",
    extracurr: "none",
    judcon: "none"
}

function Details() {
    return (
        <div className="flex flex-row w-full">
            <Sidebar />
            <div className="p-6 w-full">
                <h1 className="font-bold text-2xl mb-6">Student Details</h1>
                <div className="overflow-x-auto">
                    <table className="table-auto border-collapse border-gray-400 w-full">
                        <tbody className="bg-[#F5F4E8]">
                            <tr><td className="p-2 font-semibold">Name:</td><td className="p-2">{detailss.name}</td></tr>
                            <tr><td className="p-2 font-semibold">Roll No:</td><td className="p-2">{detailss.rollno}</td></tr>
                            <tr><td className="p-2 font-semibold">Date of Birth:</td><td className="p-2">{detailss.dob}</td></tr>
                            <tr><td className="p-2 font-semibold">Gender:</td><td className="p-2">{detailss.gender}</td></tr>
                            <tr><td className="p-2 font-semibold">Nationality:</td><td className="p-2">{detailss.nation}</td></tr>
                            <tr><td className="p-2 font-semibold">Contact:</td><td className="p-2">{detailss.contact}</td></tr>
                            <tr><td className="p-2 font-semibold">Emergency Contact:</td><td className="p-2">{detailss.emcont}</td></tr>
                        </tbody>

                        <h1 className="font-bold text-xl mb-4 mt-6">Academic Details</h1>

                        <tbody className="bg-[#F5F4E8]">
                            <tr><td className="p-2 font-semibold">Program:</td><td className="p-2">{detailss.program}</td></tr>
                            <tr><td className="p-2 font-semibold">Academic History:</td><td className="p-2">{detailss.acahist}</td></tr>
                            <tr><td className="p-2 font-semibold">Enrollment Status:</td><td className="p-2">{detailss.enrostat}</td></tr>
                            <tr><td className="p-2 font-semibold">Academic Advisor:</td><td className="p-2">{detailss.acaadvi}</td></tr>
                            <tr><td className="p-2 font-semibold">GPA:</td><td className="p-2">{detailss.gragpa}</td></tr>
                            <tr><td className="p-2 font-semibold">Academic Standing:</td><td className="p-2">{detailss.acastan}</td></tr>
                        </tbody>

                        <h1 className="font-bold text-xl mb-4 mt-6">Academic Details</h1>

                        <tbody className="bg-[#F5F4E8]">
                            <tr><td className="p-2 font-semibold">Student Credentials:</td><td className="p-2">{detailss.studcred}</td></tr>
                            <tr><td className="p-2 font-semibold">Library Access:</td><td className="p-2">{detailss.library}</td></tr>
                            <tr><td className="p-2 font-semibold">Hostel:</td><td className="p-2">{detailss.hostel}</td></tr>
                            <tr><td className="p-2 font-semibold">Hostel No:</td><td className="p-2">{detailss.hostno}</td></tr>

                        </tbody>

                        <h1 className="font-bold text-xl mb-4 mt-6">Academic Details</h1>

                        <tbody className="bg-[#F5F4E8]">
                            <tr><td className="p-2 font-semibold">Health Record:</td><td className="p-2">{detailss.healthrec}</td></tr>
                            <tr><td className="p-2 font-semibold">Disability:</td><td className="p-2">{detailss.disable}</td></tr>
                            <tr><td className="p-2 font-semibold">Extra Curricular:</td><td className="p-2">{detailss.extracurr}</td></tr>
                            <tr><td className="p-2 font-semibold">Judicial Concerns:</td><td className="p-2">{detailss.judcon}</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    )
}

export default Details;
