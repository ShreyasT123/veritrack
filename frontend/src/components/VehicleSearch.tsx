import { Search, Route } from "lucide-react";
import { useState } from "react";
export function VehicleSearch({onSearch}:{onSearch:(plate:string)=>void}) { const [plate,setPlate]=useState("GX15 OGJ"); return <section className="search-card"><label htmlFor="plate">Vehicle / HSRP query</label><div className="search-line"><Search size={17}/><input id="plate" value={plate} onChange={e=>setPlate(e.target.value.toUpperCase())}/></div><button onClick={()=>onSearch(plate)}><Route size={16}/>Reconstruct Trajectory</button><small>Warrant-scoped lookup · DPDP Act ledger entry created</small></section> }
