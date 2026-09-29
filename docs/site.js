// Publication links are enabled only after the publisher verifies the repository and assets.
fetch("publication.json").then(r=>{if(!r.ok)throw Error("unpublished");return r.json()}).then(p=>{
 if(p.status!=="published"||!/^https:\/\/github\.com\/[\w.-]+\/[\w.-]+$/.test(p.repository_url))return;
 document.querySelectorAll("[data-repository]").forEach(a=>{a.href=p.repository_url;a.hidden=false});
 document.querySelectorAll("[data-release]").forEach(a=>{a.href=p.repository_url+"/releases/tag/"+p.release_tag;a.hidden=false});
 document.getElementById("publication-status").textContent="Open research release · "+p.release_tag+" · Metadata, numerical results, code and trained adapters";
}).catch(()=>{});
