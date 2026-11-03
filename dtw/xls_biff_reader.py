   
from __future__ import annotations
import os, struct, math
from pathlib import Path
END=0xFFFFFFFE
FREE=0xFFFFFFFF
FAT=0xFFFFFFFD
DIF=0xFFFFFFFC

class CFB:
    def __init__(self, filename):
        self.buf=Path(filename).read_bytes()
        buf=self.buf
        if buf[:8]!=bytes.fromhex('D0CF11E0A1B11AE1'):
            raise ValueError('Not an OLE compound file')
        u16=lambda off:struct.unpack_from('<H',buf,off)[0]
        u32=lambda off:struct.unpack_from('<I',buf,off)[0]
        self.sector_size=1<<u16(30)
        self.mini_sector_size=1<<u16(32)
        self.mini_cutoff=u32(56)
        num_fat=u32(44)
        self.dir_start=u32(48)
        mini_fat_start=u32(60)
        num_minifat=u32(64)
        difat_start=u32(68)
        num_difat=u32(72)
        difat=list(struct.unpack_from('<109I',buf,76))
        visited=set()
        current=difat_start
        for _ in range(num_difat):
            if current in (END, FREE) or current in visited:break
            visited.add(current)
            sec=self.sector(current)
            xs=struct.unpack('<%dI'%(self.sector_size//4),sec)
            difat.extend(xs[:-1]);current=xs[-1]
        difat=[x for x in difat if x not in (FREE,END,FAT,DIF)][:num_fat]
        self.fat=[]
        for fid in difat:
            self.fat.extend(struct.unpack('<%dI'%(self.sector_size//4),self.sector(fid)))
        dbytes=self.chain_bytes(self.dir_start,self.fat)
        entries={}
        for idx in range(0,len(dbytes),128):
            rec=dbytes[idx:idx+128]
            if len(rec)<128:continue
            nl=struct.unpack_from('<H',rec,64)[0]
            ty=rec[66]
            if not nl or not ty:continue
            name=rec[:nl-2].decode('utf-16le',errors='replace')
            start=struct.unpack_from('<I',rec,116)[0]
            size=struct.unpack_from('<Q',rec,120)[0]
            entries[name]={'type':ty,'start':start,'size':size}
        self.entries=entries
        root=next((v for v in entries.values() if v['type']==5),None)
        self.mini_stream=self.chain_bytes(root['start'],self.fat)[:root['size']] if root else b''
        self.mini_fat=[]
        if num_minifat and mini_fat_start not in (END,FREE):
            minibytes=self.chain_bytes(mini_fat_start,self.fat)[:num_minifat*self.sector_size]
            self.mini_fat=list(struct.unpack('<%dI'%(len(minibytes)//4), minibytes))
    def sector(self,idx):
        i=512+idx*self.sector_size
        if i+self.sector_size > len(self.buf):raise ValueError('invalid sector')
        return self.buf[i:i+self.sector_size]
    def chain_bytes(self,start,fat):
        if start in (END,FREE):return b''
        res=[];visited=set();sid=start
        while sid not in (END,FREE,FAT,DIF):
            if sid>=len(fat) or sid in visited:raise ValueError('Bad FAT chain')
            visited.add(sid)
            res.append(self.sector(sid))
            sid=fat[sid]
        return b''.join(res)
    def get_stream(self,name):
        info=self.entries[name]
        if info['size'] >= self.mini_cutoff or info['type']==5:
            return self.chain_bytes(info['start'],self.fat)[:info['size']]
        out=[];i=info['start'];seen=set()
        while i not in (END, FREE):
            if i in seen or i>=len(self.mini_fat):raise ValueError('Bad mini-FAT chain')
            seen.add(i)
            off=i*self.mini_sector_size
            out.append(self.mini_stream[off:off+self.mini_sector_size])
            i=self.mini_fat[i]
        return b''.join(out)[:info['size']]


def rec_iter(buf, start=0, stop=None):
    pos=start
    if stop is None:stop=len(buf)
    while pos+4<=stop:
        recid, sz=struct.unpack_from('<HH',buf,pos)
        if pos+4+sz>len(buf):break
        yield pos,recid,buf[pos+4:pos+4+sz]
        pos+=4+sz
        if recid==0x000A:break             


def dec_unicode(buf,off=0,n=None):
    if n is None:
        n=struct.unpack_from('<H',buf,off)[0];off+=2
    opts=buf[off];off+=1
    rich=struct.unpack_from('<H',buf,off)[0] if opts&0x8 else 0
    if opts&0x8:off+=2
    ext=struct.unpack_from('<I',buf,off)[0] if opts&0x4 else 0
    if opts&0x4:off+=4
    blen=n*(2 if opts&1 else 1)
    string=buf[off:off+blen].decode('utf-16le' if opts&1 else 'latin-1',errors='replace')
    return string,off+blen+rich*4+ext


def parse_sst_record(chunks):
    if not chunks:return []
    data=chunks[0]
    total, unique=struct.unpack_from('<II',data,0)
                                                                                         
                                                                                        
                                             
    offset=8;current=0;vals=[]
    for _ in range(unique):
        while current<len(chunks) and offset>=len(chunks[current]):
            current+=1;offset=0
        if current>=len(chunks):break
        chunk=chunks[current]
        if offset+3>len(chunk):
            current+=1;offset=0;chunk=chunks[current]
        n=struct.unpack_from('<H',chunk,offset)[0]
        flags=chunk[offset+2]
        offset+=3
        rich=0;ext=0
        if flags&0x8:
            rich=struct.unpack_from('<H',chunk,offset)[0];offset+=2
        if flags&0x4:
            ext=struct.unpack_from('<I',chunk,offset)[0];offset+=4
        unit=2 if flags&1 else 1
        needed=n*unit
        char_bytes=[]
        while needed>0:
            chunk=chunks[current]
            avail=len(chunk)-offset
            read=min(needed,avail)
            if read>0:char_bytes.append(chunk[offset:offset+read]);offset+=read;needed-=read
            if needed:
                current+=1;offset=0
                if current>=len(chunks):raise ValueError('Incomplete SST')
                                                                                      
                new_mode=chunks[current][0];offset=1
                if (new_mode&1)!=(flags&1):
                    raise ValueError('SST continuation encoding changes, not supported')
        val=b''.join(char_bytes).decode('utf-16le' if flags&1 else 'latin-1',errors='replace')
        vals.append(val)
        remaining=rich*4+ext
        while remaining:
            chunk=chunks[current];eat=min(remaining,len(chunk)-offset)
            offset+=eat;remaining-=eat
            if remaining:
                current+=1;offset=0
    return vals


def unpack_rk(raw):
    if raw&2:
        v=(raw>>2)
        if v&(1<<29):v-=(1<<30)
    else:
        v=struct.unpack('<d',struct.pack('<Q',(raw&0xFFFFFFFC)<<32))[0]
    if raw&1:v/=100.0
    return v


def read_xls(file):
    cfb=CFB(file)
    name=next((x for x in ['Workbook','Book'] if x in cfb.entries),None)
    if not name:raise ValueError('Missing workbook stream')
    stream=cfb.get_stream(name)
    globals=[]
    sheets=[]
    sst_chunks=[];is_sst=False
    for pos,code,d in rec_iter(stream):
        globals.append((pos,code,len(d)))
        if code==0x0085:
            off=struct.unpack_from('<I',d,0)[0]
            n=d[6];fl=d[7]
            name=d[8:8+n*(2 if fl&1 else 1)].decode('utf-16le' if fl&1 else 'latin-1',errors='replace')
            sheets.append((name,off))
        if code==0x00FC:
            sst_chunks=[d];is_sst=True
        elif code==0x003C and is_sst:
            sst_chunks.append(d)
        elif code!=0x003C:
            is_sst=False
    sst=parse_sst_record(sst_chunks)
    all_sheets={}
    for title, off in sheets:
        cells={}
        for pos,code,d in rec_iter(stream,off):
            if code==0x00FD:
                r,c,xf,index=struct.unpack_from('<HHHI',d,0)
                cells[(r,c)]=sst[index] if index<len(sst) else f'#BadSST{index}'
            elif code==0x0203:
                r,c,xf=struct.unpack_from('<HHH',d,0)
                cells[(r,c)]=struct.unpack_from('<d',d,6)[0]
            elif code==0x027E:
                r,c,xf,v=struct.unpack_from('<HHHI',d,0)
                cells[(r,c)]=unpack_rk(v)
            elif code==0x00BD:
                r,first=struct.unpack_from('<HH',d,0)
                last=struct.unpack_from('<H',d,len(d)-2)[0]
                for j,c in enumerate(range(first,last+1)):
                    xf,v=struct.unpack_from('<HI',d,4+j*6)
                    cells[(r,c)]=unpack_rk(v)
            elif code==0x0204:
                r,c,xf=struct.unpack_from('<HHH',d,0)
                try:cells[(r,c)]=dec_unicode(d,6)[0]
                except:cells[(r,c)]=d[8:].decode('latin-1',errors='replace')
            elif code==0x0006:
                r,c,xf=struct.unpack_from('<HHH',d,0)
                result=d[6:14]
                if result[6:]!=b'\xff\xff':
                    cells[(r,c)]=struct.unpack('<d',result)[0]
            elif code==0x0205:
                r,c,xf,flag,err=struct.unpack_from('<HHHBB',d,0)
                cells[(r,c)]=bool(flag) if err==0 else None
        if cells:
            mr=max(r for r,c in cells);mc=max(c for r,c in cells)
            matrix=[[cells.get((r,c)) for c in range(mc+1)] for r in range(mr+1)]
        else:matrix=[]
        all_sheets[title]=matrix
    return all_sheets

if __name__=='__main__':
    import sys
    for f in sys.argv[1:]:
        print('\n#####', f)
        sheets=read_xls(f)
        for name, rows in sheets.items():
            print('SHEET',repr(name),'shape',len(rows),'x',max(map(len,rows),default=0))
            print('top:', *[str(r[:10]) for r in rows[:7]], sep='\n  ')
