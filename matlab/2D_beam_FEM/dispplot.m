% file: dispplot.m


DispDisplayRatio=200;

%ConnectOfElem=[1,5,2;
%               2,6,3;
%               3,7,4;
%               4,8,1
%              ];
ConnectOfElem=[1,0,2;2,0,3;3,0,4;4,0,1];
nEdgePerElem=4;
               
	% cal the total lines
TotalLines=0;

for ie=1:numele
    for iedge=1:nEdgePerElem
        if( ConnectOfElem(iedge,2) ~= 0)   %there exist a mid-node
          inode2=node(ConnectOfElem(iedge,2),ie);
          TotalLines=TotalLines+2;
        else	                    % there is no mid-node
           TotalLines=TotalLines+1;
        end       % endif
    end % endfor iedge
end   % endfor ie

iSparseIndex=zeros(TotalLines,1);
jSparseIndex=zeros(TotalLines,1);

index=0;
for ie=1:numele               
    for iedge=1:nEdgePerElem
        inode1=node(ConnectOfElem(iedge,1),ie);
        inode3=node(ConnectOfElem(iedge,3),ie);

	if ( ConnectOfElem(iedge,2) ~= 0 ) % there exist a mid-node
           inode2=node(ConnectOfElem(iedge,2),ie);                
           index=index+1;
           iSparseIndex(index)=inode1;
           jSparseIndex(index)=inode2;
           index=index+1;
           iSparseIndex(index)=inode2;
           jSparseIndex(index)=inode3;
        else	                    % there is no mid-node
           index=index+1;
           iSparseIndex(index)=inode1;
           jSparseIndex(index)=inode3;
        end     % endif
    end    % endfor idege
end   % endfor ie

Spar = sparse(iSparseIndex,jSparseIndex,ones(1,TotalLines));

figure(1)
%gplot(Spar,[coord(1,:)' coord(2,:)'])

    % display the circle on the node ( the undeformed )
%hold on
%gplot(eye(numnod),coord','o')


	% coordAfterDeform  is   2  by  2*numnod matrix
coordAfterDeform=[(coord(1,:)+DispDisplayRatio.*disp(1:2:2*numnod-1));
                  (coord(2,:)+DispDisplayRatio.*disp(2:2:2*numnod))  ]; 


hold on
gplot(Spar,[coordAfterDeform(1,:)' coordAfterDeform(2,:)'],'-.')

   % display the star on the node ( deformation )
%hold on
%gplot(eye(numnod),coordAfterDeform','*')

title(['Deformation and Analytical solution']);
axis equal;

	% plot the mesh and the node
%figure(2);
%gplot(Spar,[coord(1,:)' coord(2,:)'])
%hold on
%gplot(eye(numnod),coord','o')
%title(['Mesh ',int2str(numele),' elements']);
hold off;
axis equal;











